use melior::{
    dialect::{qwerty, DialectHandle, DialectRegistry},
    ir::{Block, Value},
    utility::register_inliner_extensions,
    Context,
};
use qwerty_ast::typecheck::TypeEnv;
use std::{collections::HashMap, sync::LazyLock};

/// Holds the MLIR context in static memory, initializing it on first use.
pub static MLIR_CTX: LazyLock<Context> = LazyLock::new(|| {
    let ctx = Context::new();
    let registry = DialectRegistry::new();
    let dialects = [
        DialectHandle::arith(),
        DialectHandle::cf(),
        DialectHandle::scf(),
        DialectHandle::func(),
        DialectHandle::math(),
        DialectHandle::llvm(),
        DialectHandle::ccirc(),
        DialectHandle::qcirc(),
        DialectHandle::qwerty(),
    ];

    for dialect in dialects {
        dialect.insert_dialect(&registry);
    }
    register_inliner_extensions(&registry);
    ctx.append_dialect_registry(&registry);

    for dialect in dialects {
        dialect.load_dialect(&ctx);
    }

    ctx
});

/// Something that is bound to a name and can be (or has been) materialized to
/// MLIR values.
pub enum BoundVals {
    /// Already materialized
    Materialized(Vec<Value<'static, 'static>>),

    /// A function symbol name that has not yet been materialized into a
    /// `qwerty::FuncConstOp`.
    UnmaterializedFunction(qwerty::FunctionType<'static>),
}

pub struct Ctx<'a> {
    pub root_block: &'a Block<'static>,
    pub type_env: TypeEnv,
    pub bindings: HashMap<String, BoundVals>,
}

impl<'a> Ctx<'a> {
    pub fn new(root_block: &'a Block<'static>, type_env: TypeEnv) -> Self {
        Self {
            root_block,
            type_env,
            bindings: HashMap::new(),
        }
    }
}
