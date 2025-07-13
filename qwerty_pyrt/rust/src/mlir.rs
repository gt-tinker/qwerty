use melior::{
    dialect::{arith, qwerty, DialectHandle},
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, OperationLike, Region, RegionLike,
    },
    Context,
};
use qwerty_ast::{
    ast::{FunctionDef, Program},
    dbg::DebugLoc,
};
use std::sync::LazyLock;

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

fn func_mlir_ty(func_def: &FunctionDef) -> qwerty::FunctionType {
    // TODO: update by using Dinesh's new field
    let is_rev = false;
    // TODO: use actual argument types instead of hardcoding
    qwerty::FunctionType::new(
        &MLIR_CTX,
        FunctionType::new(
            &MLIR_CTX,
            &[],
            &[qwerty::BitBundleType::new(&MLIR_CTX, 1).into()],
        ),
        is_rev,
    )
}

pub fn ast_to_mlir(prog: &Program) -> Module {
    let loc = dbg_to_loc(prog.dbg.clone());
    let module = Module::new(loc);
    let module_block = module.body();

    for func in &prog.funcs {
        let sym_name = StringAttribute::new(&MLIR_CTX, &func.name);
        let func_ty_attr = TypeAttribute::new(func_mlir_ty(func).into());
        let func_attrs = &[];
        let func_loc = dbg_to_loc(prog.dbg.clone());

        // TODO: use actual arguments (currently this is no args)
        let func_block = Block::new(&[]);
        // TODO: use actual node locs
        let const0 = func_block
            .append_operation(arith::constant(
                &MLIR_CTX,
                IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 1).into(), 0).into(),
                func_loc,
            ))
            .result(0)
            .unwrap()
            .into();
        let bundle = func_block
            .append_operation(qwerty::bitpack(&[const0], func_loc))
            .result(0)
            .unwrap()
            .into();
        func_block.append_operation(qwerty::r#return(&[bundle], func_loc));

        let func_region = Region::new();
        func_region.append_block(func_block);

        let func_op = qwerty::func(
            &MLIR_CTX,
            sym_name,
            func_ty_attr,
            func_region,
            func_attrs,
            func_loc,
        );
        module_block.append_operation(func_op);
    }

    assert!(module.as_operation().verify());
    // TODO: don't dump. use executionengine instead
    module.as_operation().dump();

    module
}
