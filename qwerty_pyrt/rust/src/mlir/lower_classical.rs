use crate::mlir::{
    ctx::{BoundVals, Ctx, MLIR_CTX},
    lower_prog_stmt::{Lowerable, ast_stmt_to_mlir},
    lower_type::{ast_ty_to_mlir_tys, dbg_to_loc},
};
use melior::{
    dialect::ccirc,
    ir::{
        self, Block, BlockLike, Location, Operation, OperationLike, Region, RegionLike, Value,
        attribute::{BoolAttribute, IntegerAttribute, StringAttribute},
        operation::OperationResult,
        symbol_table::Visibility,
        r#type::IntegerType,
    },
};
use qwerty_ast::{
    ast::{
        self, BitLiteral, FunctionDef, Variable,
        classical::{self, BinaryOp, BinaryOpKind, ReduceOp, Slice, UnaryOp, UnaryOpKind, ModMul},
    },
    typecheck::{ComputeKind, FuncsAvailable},
};

impl Lowerable for classical::Expr {
    fn lower_to_mlir(
        &self,
        ctx: &mut Ctx,
        block: &Block<'static>,
    ) -> (ast::Type, ComputeKind, Vec<Value<'static, 'static>>) {
        ast_classical_expr_to_mlir(self, ctx, block)
    }

    fn build_return(
        vals: &[Value<'static, 'static>],
        loc: Location<'static>,
    ) -> Operation<'static> {
        ccirc::r#return(vals, loc)
    }

    fn bit_register_ty(dim: usize) -> ir::Type<'static> {
        ccirc::WireType::new(&MLIR_CTX, dim.try_into().unwrap()).into()
    }

    fn build_bit_register_unpack(
        reg: Value<'static, 'static>,
        loc: Location<'static>,
    ) -> Operation<'static> {
        ccirc::wireunpack(reg, loc)
    }
}

/// Converts a `@classical` AST Expr node to mlir::Values by appending ops to
/// the provided block.
fn ast_classical_expr_to_mlir(
    expr: &classical::Expr,
    ctx: &mut Ctx,
    block: &Block<'static>,
) -> (ast::Type, ComputeKind, Vec<Value<'static, 'static>>) {
    match expr {
        classical::Expr::Variable(var @ Variable { name, .. }) => {
            let (ty, compute_kind) = var
                .calc_type(&mut ctx.type_env)
                .expect("Variable to pass typechecking");
            let bound_vals = ctx
                .bindings
                .get(name)
                .expect(&format!("Variable {} to be bound", name));

            let mlir_vals = match bound_vals {
                BoundVals::Materialized(vals) => vals.clone(),
                BoundVals::UnmaterializedFunction(_) => {
                    panic!("@classical variables should never require materialization")
                }
            };

            (ty, compute_kind, mlir_vals)
        }

        classical::Expr::Slice(
            slice @ Slice {
                val,
                lower,
                upper,
                dbg,
            },
        ) => {
            let (val_ty, val_compute_kind, val_vals) =
                ast_classical_expr_to_mlir(&**val, ctx, block);
            assert_eq!(val_vals.len(), 1, "wire should have 1 mlir value");
            let val_val = val_vals[0];

            let (ty, compute_kind) = slice
                .calc_type(&(val_ty, val_compute_kind))
                .expect("Slice to pass type checking");
            let loc = dbg_to_loc(dbg.clone());

            assert!(*upper >= *lower);
            let sliced_vals: Vec<_> = block
                .append_operation(ccirc::wireunpack(val_val, loc))
                .results()
                .map(OperationResult::into)
                .skip(*lower)
                .take(*upper - *lower)
                .collect();
            let mlir_vals = vec![
                block
                    .append_operation(ccirc::wirepack(&sliced_vals, loc))
                    .result(0)
                    .unwrap()
                    .into(),
            ];
            (ty, compute_kind, mlir_vals)
        }

        classical::Expr::UnaryOp(unary @ UnaryOp { kind, val, dbg }) => {
            let (val_ty, val_compute_kind, val_vals) =
                ast_classical_expr_to_mlir(&**val, ctx, block);
            assert_eq!(val_vals.len(), 1, "wire should have 1 mlir value");
            let val_val = val_vals[0];

            let (ty, compute_kind) = unary
                .calc_type(&(val_ty, val_compute_kind))
                .expect("UnaryOp to pass type checking");

            let loc = dbg_to_loc(dbg.clone());
            let op = match kind {
                UnaryOpKind::Not => ccirc::not(val_val, loc),
            };
            let mlir_vals = vec![block.append_operation(op).result(0).unwrap().into()];
            (ty, compute_kind, mlir_vals)
        }

        classical::Expr::BinaryOp(
            binary @ BinaryOp {
                kind,
                left,
                right,
                dbg,
            },
        ) => {
            let (left_ty, left_compute_kind, left_vals) =
                ast_classical_expr_to_mlir(&**left, ctx, block);
            assert_eq!(left_vals.len(), 1, "wire should have 1 mlir value");
            let left_val = left_vals[0];

            let (right_ty, right_compute_kind, right_vals) =
                ast_classical_expr_to_mlir(&**right, ctx, block);
            assert_eq!(right_vals.len(), 1, "wire should have 1 mlir value");
            let right_val = right_vals[0];

            let (ty, compute_kind) = binary
                .calc_type(
                    &(left_ty, left_compute_kind),
                    &(right_ty, right_compute_kind),
                )
                .expect("BinaryOp to pass type checking");

            let loc = dbg_to_loc(dbg.clone());
            let op = match kind {
                BinaryOpKind::And => ccirc::and(left_val, right_val, loc),
                BinaryOpKind::Or => ccirc::or(left_val, right_val, loc),
                BinaryOpKind::Xor => ccirc::xor(left_val, right_val, loc),
            };
            let mlir_vals = vec![block.append_operation(op).result(0).unwrap().into()];
            (ty, compute_kind, mlir_vals)
        }

        classical::Expr::ReduceOp(reduce @ ReduceOp { kind, val, dbg }) => {
            let (val_ty, val_compute_kind, val_vals) =
                ast_classical_expr_to_mlir(&**val, ctx, block);
            assert_eq!(val_vals.len(), 1, "wire should have 1 mlir value");
            let val_val = val_vals[0];

            let (ty, compute_kind) = reduce
                .calc_type(&(val_ty, val_compute_kind))
                .expect("ReduceOp to pass type checking");

            let loc = dbg_to_loc(dbg.clone());
            let reduced_wire = block
                .append_operation(ccirc::wireunpack(val_val, loc))
                .results()
                .map(OperationResult::into)
                .reduce(|acc, wire| {
                    let op = match kind {
                        BinaryOpKind::And => ccirc::and(acc, wire, loc),
                        BinaryOpKind::Or => ccirc::or(acc, wire, loc),
                        BinaryOpKind::Xor => ccirc::xor(acc, wire, loc),
                    };
                    block.append_operation(op).result(0).unwrap().into()
                })
                .expect("A wire should never be unpacked into zero wires");
            let mlir_vals = vec![reduced_wire];
            (ty, compute_kind, mlir_vals)
        }

        classical::Expr::RotateOp(_) => todo!("@classical rotate op"),

        classical::Expr::Concat(_) => todo!("@classical concat op"),

        classical::Expr::Repeat(_) => todo!("@classical repeat op"),

        classical::Expr::ModMul(modmul @ ModMul { x, j, y, mod_n, dbg }) => {
            let (y_ty, y_compute_kind, y_vals) =
                ast_classical_expr_to_mlir(&**y, ctx, block);
            assert_eq!(y_vals.len(), 1, "wire should have 1 mlir value");
            let y_val = y_vals[0];

            let (ty, compute_kind) = modmul
                .calc_type(&(y_ty, y_compute_kind))
                .expect("ModMul to pass type checking");

            let loc = dbg_to_loc(dbg.clone());
            let x_attr = IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 64).into(), *x as i64);
            let j_attr = IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 64).into(), *j as i64);
            let mod_n_attr = IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 64).into(), *mod_n as i64);
            let product = block
                .append_operation(ccirc::modmul(&MLIR_CTX, x_attr, j_attr, mod_n_attr, y_val, loc))
                .result(0)
                .unwrap()
                .into();
            let mlir_vals = vec![product];
            (ty, compute_kind, mlir_vals)
        }

        classical::Expr::BitLiteral(blit @ BitLiteral { val, n_bits, dbg }) => {
            let (ty, compute_kind) = blit.calc_type().expect("BitLiteral to pass type checking");

            let loc = dbg_to_loc(dbg.clone());
            let value_attr =
                IntegerAttribute::from_ubig(&MLIR_CTX, (*n_bits).try_into().unwrap(), val);
            let mlir_vals = vec![
                block
                    .append_operation(ccirc::constant(&MLIR_CTX, value_attr, loc))
                    .result(0)
                    .unwrap()
                    .into(),
            ];

            (ty, compute_kind, mlir_vals)
        }
    }
}

/// Converts an AST `@classical` `FunctionDef` node into a `ccirc::circuit` op.
pub fn ast_classical_func_def_to_mlir(
    func_def: &FunctionDef<classical::Expr>,
    funcs_available: &FuncsAvailable,
) -> Operation<'static> {
    let func_loc = dbg_to_loc(func_def.dbg.clone());

    let block_args: Vec<_> = func_def
        .args
        .iter()
        .map(|(arg_ty, _arg_name)| {
            let mlir_tys = ast_ty_to_mlir_tys::<classical::Expr>(arg_ty);
            // @classical functions should take only bits
            assert_eq!(mlir_tys.len(), 1);
            (mlir_tys[0], func_loc)
        })
        .collect();
    let func_block = Block::new(&block_args);

    let type_env = func_def
        .new_type_env(funcs_available)
        .expect("valid type env");
    let mut ctx = Ctx::new(&func_block, type_env);

    // Bind function arguments
    assert_eq!(func_def.args.len(), func_block.argument_count());
    for (arg_name, arg_val) in func_def
        .args
        .iter()
        .map(|(_ty, name)| name)
        .zip(func_block.arguments())
    {
        let old_binding = ctx.bindings.insert(
            arg_name.to_string(),
            BoundVals::Materialized(vec![arg_val.into()]),
        );
        assert!(old_binding.is_none());
    }

    for stmt in &func_def.body {
        let compute_kind = ast_stmt_to_mlir(
            stmt,
            &mut ctx,
            &func_block,
            func_def.get_expected_ret_type(),
        );
        func_def
            .check_stmt_compute_kind(compute_kind)
            .expect("Statement to have a valid ComputeKind");
    }

    let func_region = Region::new();
    func_region.append_block(func_block);

    let is_rev_attr = BoolAttribute::new(&MLIR_CTX, func_def.is_rev);
    let sym_name_attr = StringAttribute::new(&MLIR_CTX, &func_def.name);

    ccirc::circuit(
        &MLIR_CTX,
        is_rev_attr,
        sym_name_attr,
        // Can't be called from Python
        Visibility::Private,
        func_region,
        func_loc,
    )
}
