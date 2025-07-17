mod mlir;
mod wrap_ast;
mod wrap_repl;

use crate::wrap_ast::{
    Basis, DebugLoc, Expr, FunctionDef, Program, QLit, RegKind, Stmt, Type, Vector,
};
use crate::wrap_repl::{ReplState, TypeEnv};
use pyo3::prelude::*;

/// The Python extension module allowing the Python portion of the Qwerty
/// runtime to communicate with the Rust and C++ portions of the runtime.
/// Following the convention in CPython, we begin the name of the extension
/// module with an underscore.
#[pymodule]
#[pyo3(name = "_qwerty_pyrt")]
fn qwerty_pyrt(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // AST wrapper classes
    module.add_class::<DebugLoc>()?;
    module.add_class::<RegKind>()?;
    module.add_class::<Type>()?;
    module.add_class::<QLit>()?;
    module.add_class::<Vector>()?;
    module.add_class::<Basis>()?;
    module.add_class::<Expr>()?;
    module.add_class::<Stmt>()?;
    module.add_class::<FunctionDef>()?;
    module.add_class::<Program>()?;
    module.add_class::<ReplState>()?;
    module.add_class::<TypeEnv>()?;
    module.add_class::<Type>()?;

    Ok(())
}
