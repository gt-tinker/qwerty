use melior::{
    dialect::DialectHandle,
    ir::{Location, Module},
    Context,
};
use qwerty_ast::{ast::Program, dbg::DebugLoc};
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

pub fn ast_to_mlir(prog: &Program) -> Module {
    let loc = dbg_to_loc(prog.dbg.clone());
    let module = Module::new(loc);

    // TODO: lower ast

    module
}
