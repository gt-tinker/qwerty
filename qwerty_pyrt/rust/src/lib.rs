mod mlir;
mod wrap_ast;
mod wrap_repl;

use crate::wrap_ast::{
    Basis, BasisGenerator, ClassicalExpr, ClassicalFunctionDef, ClassicalStmt, DebugLoc, Program,
    QLit, QpuExpr, QpuFunctionDef, QpuStmt, RegKind, Type, TypeEnv, UnaryOpKind, Vector,
};
use crate::wrap_repl::ReplState;
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
    module.add_class::<BasisGenerator>()?;
    module.add_class::<Basis>()?;
    module.add_class::<QpuExpr>()?;
    module.add_class::<ClassicalExpr>()?;
    module.add_class::<UnaryOpKind>()?;
    module.add_class::<QpuStmt>()?;
    module.add_class::<ClassicalStmt>()?;
    module.add_class::<QpuFunctionDef>()?;
    module.add_class::<ClassicalFunctionDef>()?;
    module.add_class::<Program>()?;
    module.add_class::<ReplState>()?;
    module.add_class::<TypeEnv>()?;

    Ok(())
}
