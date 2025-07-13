mod wrap_ast;

use pyo3::prelude::*;
use wrap_ast::{Basis, DebugLoc, Expr, FunctionDef, Program, QLit, RegKind, Stmt, Type, Vector};

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

    Ok(())
}
