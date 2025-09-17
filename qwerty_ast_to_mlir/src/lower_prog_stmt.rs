use crate::{
    ctx::{BoundVals, Ctx},
    lower_classical::ast_classical_func_def_to_mlir,
    lower_qpu::ast_qpu_func_def_to_mlir,
    lower_type::dbg_to_loc,
};
use melior::{
    dialect::qwerty,
    ir::{self, Block, BlockLike, Location, Module, Operation, OperationLike, Value},
};
use qwerty_ast::{
    ast::{self, Assign, Func, Program, RegKind, Return, Stmt, UnpackAssign},
    typecheck::{ComputeKind, FuncsAvailable, TypeCheckable},
};

/// This trait is a hack to allow calling either [`ast_qpu_expr_to_mlir`] or
/// [`ast_classical_expr_to_mlir`] from [`ast_stmt_to_mlir`] as appropriate
/// depending on the type of `FunctionDef` involved.
pub trait Lowerable {
    fn lower_to_mlir(
        &self,
        ctx: &mut Ctx,
        block: &Block<'static>,
    ) -> (ast::Type, ComputeKind, Vec<Value<'static, 'static>>);

    fn build_return(vals: &[Value<'static, 'static>], loc: Location<'static>)
    -> Operation<'static>;

    fn bit_register_ty(dim: usize) -> ir::Type<'static>;

    fn build_bit_register_unpack(
        reg: Value<'static, 'static>,
        loc: Location<'static>,
    ) -> Operation<'static>;
}

/// Append ops that implement an AST Stmt node to the provided block.
pub fn ast_stmt_to_mlir<E>(
    stmt: &Stmt<E>,
    ctx: &mut Ctx,
    block: &Block<'static>,
    expected_ret_type: Option<ast::Type>,
) -> ComputeKind
where
    E: Lowerable + TypeCheckable,
{
    match stmt {
        Stmt::Expr(expr) => {
            let (_ty, compute_kind, _vals) = expr.expr.lower_to_mlir(ctx, block);
            compute_kind
        }

        Stmt::Assign(assign @ Assign { lhs, rhs, .. }) => {
            let (rhs_type, rhs_compute_kind, rhs_vals) = rhs.lower_to_mlir(ctx, block);
            ctx.bindings
                .insert(lhs.to_string(), BoundVals::Materialized(rhs_vals));
            assign
                .finish_type_checking(&mut ctx.type_env, &(rhs_type, rhs_compute_kind))
                .expect("Assign to finish typechecking")
        }

        Stmt::UnpackAssign(unpack @ UnpackAssign { lhs, rhs, dbg }) => {
            let (rhs_ty, rhs_compute_kind, rhs_vals) = rhs.lower_to_mlir(ctx, block);
            let loc = dbg_to_loc(dbg.clone());

            assert!(!lhs.is_empty());
            assert_eq!(rhs_vals.len(), 1);
            let rhs_val = rhs_vals[0];

            let unpacked_vals: Vec<Value<'static, 'static>> = match rhs_ty {
                ast::Type::RegType {
                    elem_ty: RegKind::Qubit,
                    ..
                } => block
                    .append_operation(qwerty::qbunpack(rhs_val, loc))
                    .results()
                    .map(|res| {
                        block
                            .append_operation(qwerty::qbpack(&[res.into()], loc))
                            .result(0)
                            .unwrap()
                            .into()
                    })
                    .collect(),

                ast::Type::RegType {
                    elem_ty: RegKind::Bit,
                    ..
                } => block
                    .append_operation(E::build_bit_register_unpack(rhs_val, loc))
                    .results()
                    .map(|res| {
                        block
                            .append_operation(qwerty::bitpack(&[res.into()], loc))
                            .result(0)
                            .unwrap()
                            .into()
                    })
                    .collect(),

                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    ..
                } => panic!("cannot unpack basis"),

                ast::Type::FuncType { .. }
                | ast::Type::RevFuncType { .. }
                | ast::Type::TupleType { .. }
                | ast::Type::UnitType => panic!("Can only unpack registers"),
            };

            assert_eq!(lhs.len(), unpacked_vals.len());
            for (lhs_name, rhs_val) in lhs.iter().zip(unpacked_vals.iter()) {
                ctx.bindings.insert(
                    lhs_name.to_string(),
                    BoundVals::Materialized(vec![*rhs_val]),
                );
            }

            unpack
                .finish_type_checking(&mut ctx.type_env, &(rhs_ty, rhs_compute_kind))
                .expect("UnpackAssign to finish typechecking")
        }

        Stmt::Return(ret @ Return { val, dbg }) => {
            let (val_ty, val_compute_kind, vals) = val.lower_to_mlir(ctx, block);
            let loc = dbg_to_loc(dbg.clone());
            block.append_operation(E::build_return(&vals, loc));

            ret.finish_type_checking(&(val_ty, val_compute_kind), expected_ret_type)
                .expect("Return to finish typechecking")
        }
    }
}

/// Converts a Qwerty AST into an mlir::ModuleOp.
pub fn ast_program_to_mlir(prog: &Program) -> Module<'static> {
    let loc = dbg_to_loc(prog.dbg.clone());
    let module = Module::new(loc);
    let module_block = module.body();
    let mut funcs_available = FuncsAvailable::new();
    let mut mlir_func_tys = vec![];

    for func in &prog.funcs {
        let (func_op, mlir_func_ty_opt) = match func {
            Func::Qpu(func_def) => {
                funcs_available.add_qpu_kernel(func_def);
                let (func_op, mlir_func_ty) =
                    ast_qpu_func_def_to_mlir(func_def, &funcs_available, &mlir_func_tys);
                (func_op, Some(mlir_func_ty))
            }
            Func::Classical(func_def) => {
                funcs_available.add_classical_func(func_def);
                let func_op = ast_classical_func_def_to_mlir(func_def, &funcs_available);
                (func_op, None)
            }
        };

        module_block.append_operation(func_op);
        if let Some(mlir_func_ty) = mlir_func_ty_opt {
            mlir_func_tys.push((func.get_name(), mlir_func_ty));
        }
    }

    assert!(module.as_operation().verify());

    module
}
