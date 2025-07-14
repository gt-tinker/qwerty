use melior::{
    dialect::{arith, qwerty, DialectHandle},
    ir::{
        self,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, Operation, OperationLike, Region, RegionLike, Value
    },
    Context,
};
use qwerty_ast::{
    ast::{self, FunctionDef, Program, RegKind, Expr, Stmt},
    dbg::DebugLoc,
};
use std::{sync::LazyLock, collections::HashMap};

static MLIR_CTX: LazyLock<Context> = LazyLock::new(|| {
    let ctx = Context::new();

    for dialect in [
        DialectHandle::arith(),
        DialectHandle::cf(),
        DialectHandle::scf(),
        DialectHandle::func(),
        DialectHandle::math(),
        DialectHandle::llvm(),
        DialectHandle::qcirc(),
        DialectHandle::qwerty(),
    ] {
        dialect.register_dialect(&ctx);
        dialect.load_dialect(&ctx);
    }

    ctx
});

fn dbg_to_loc(dbg: Option<DebugLoc>) -> Location<'static> {
    dbg.map_or_else(
        || Location::unknown(&MLIR_CTX),
        |dbg| Location::new(&MLIR_CTX, &dbg.file, dbg.line, dbg.col),
    )
}

fn ast_ty_to_mlir_tys(ty: &ast::Type) -> Vec<ir::Type<'static>> {
    match ty {
        ast::Type::FuncType { in_ty, out_ty } => {
            vec![qwerty::FunctionType::new(
                &MLIR_CTX,
                FunctionType::new(
                    &MLIR_CTX,
                    &ast_ty_to_mlir_tys(&**in_ty),
                    &ast_ty_to_mlir_tys(&**out_ty),
                ),
                /*reversible=*/ false,
            )
            .into()]
        }

        ast::Type::RevFuncType { in_out_ty } => {
            let in_out_mlir_tys = ast_ty_to_mlir_tys(&**in_out_ty);
            vec![qwerty::FunctionType::new(
                &MLIR_CTX,
                FunctionType::new(&MLIR_CTX, &in_out_mlir_tys, &in_out_mlir_tys),
                /*reversible=*/ true,
            )
            .into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } => {
            vec![qwerty::BitBundleType::new(&MLIR_CTX, *dim).into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Qubit,
            dim,
        } => {
            vec![qwerty::QBundleType::new(&MLIR_CTX, *dim).into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Basis,
            ..
        } => unreachable!(),

        ast::Type::UnitType => vec![],
        // TODO: Support TupleType once added to qwery_ast::ast::Type (that is
        //       the reason why this returns a Vec)
    }
}

fn ast_func_mlir_ty(func_def: &FunctionDef) -> ir::Type<'static> {
    let mlir_tys = ast_ty_to_mlir_tys(&func_def.get_type());
    assert_eq!(mlir_tys.len(), 1);
    mlir_tys[0]
}

struct Ctx {
    bindings: HashMap<String, Vec<Value<'static, 'static>>>,
}

impl Ctx {
    fn new() -> Self {
        Self { bindings: HashMap::new() }
    }
}

fn ast_expr_to_mlir(expr: &Expr, ctx: &Ctx, block: &Block) -> Vec<Value<'static, 'static>> {
    match expr {
        // TODO: left off here
        Expr::QLit { qlit, dbg } => todo!(),
        _ => todo!(),
    }
}

fn ast_stmt_to_mlir(stmt: &Stmt, ctx: &mut Ctx, block: &Block) {
    match stmt {
        Stmt::Expr { expr, dbg: _ } => {
            ast_expr_to_mlir(expr, ctx, block);
        }

        Stmt::Assign { lhs, rhs, dbg: _ } => {
            let rhs_vals = ast_expr_to_mlir(rhs, ctx, block);
            ctx.bindings.insert(lhs.to_string(), rhs_vals);
        }

        // TODO: for now, unpacking bundles. for tomorrow, unpacking tuples too
        Stmt::UnpackAssign { lhs, rhs, dbg: _ } => todo!(),

        Stmt::Return { val, dbg } => {
            let vals = ast_expr_to_mlir(val, ctx, block);
            let loc = dbg_to_loc(dbg.clone());
            block.append_operation(qwerty::r#return(&vals, loc));
        }
    }
}

pub fn ast_func_def_to_mlir(func_def: &FunctionDef) -> Operation<'static> {
    let sym_name = StringAttribute::new(&MLIR_CTX, &func_def.name);
    let func_ty = ast_func_mlir_ty(func_def);
    let func_ty_attr = TypeAttribute::new(func_ty);
    let func_attrs = &[];
    let func_loc = dbg_to_loc(func_def.dbg.clone());

    let qwerty_func_ty: qwerty::FunctionType = func_ty.try_into().unwrap();
    let block_args: Vec<_> = qwerty_func_ty.get_function_type().inputs().iter().map(|ty| (*ty, func_loc)).collect();
    let func_block = Block::new(&block_args);

    let mut ctx = Ctx::new();
    for stmt in &func_def.body {
        ast_stmt_to_mlir(stmt, &mut ctx, &func_block);
    }

    // TODO: use actual node locs
    //let const0 = func_block
    //    .append_operation(arith::constant(
    //        &MLIR_CTX,
    //        IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 1).into(), 0).into(),
    //        func_loc,
    //    ))
    //    .result(0)
    //    .unwrap()
    //    .into();
    //let bundle = func_block
    //    .append_operation(qwerty::bitpack(&[const0], func_loc))
    //    .result(0)
    //    .unwrap()
    //    .into();
    //func_block.append_operation(qwerty::r#return(&[bundle], func_loc));

    let func_region = Region::new();
    func_region.append_block(func_block);

    qwerty::func(
        &MLIR_CTX,
        sym_name,
        func_ty_attr,
        func_region,
        func_attrs,
        func_loc,
    )
}

pub fn ast_program_to_mlir(prog: &Program) -> Module {
    let loc = dbg_to_loc(prog.dbg.clone());
    let module = Module::new(loc);
    let module_block = module.body();

    for func_def in &prog.funcs {
        let func_op = ast_func_def_to_mlir(func_def);
        module_block.append_operation(func_op);
    }

    assert!(module.as_operation().verify());
    // TODO: don't dump. use executionengine instead
    module.as_operation().dump();

    module
}
