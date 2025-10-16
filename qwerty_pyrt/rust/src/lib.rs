mod wrap_ast;
mod wrap_repl;

use crate::wrap_ast::{
    Backend, Basis, BasisGenerator, BasisMacroPattern, BinaryOpKind, ClassicalExpr,
    ClassicalFunctionDef, ClassicalStmt, DebugLoc, DimExpr, DimVar, EmbedKind, ExprMacroPattern,
    FloatExpr, MacroEnv, PlainQpuExpr, PlainQpuStmt, Program, QpuExpr, QpuFunctionDef, QpuPrelude,
    QpuStmt, RecDefParam, RegKind, Type, TypeEnv, UnaryOpKind, Vector,
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
    module.add_class::<FloatExpr>()?;
    module.add_class::<Vector>()?;
    module.add_class::<BasisGenerator>()?;
    module.add_class::<Basis>()?;
    module.add_class::<QpuExpr>()?;
    module.add_class::<EmbedKind>()?;
    module.add_class::<ClassicalExpr>()?;
    module.add_class::<UnaryOpKind>()?;
    module.add_class::<BinaryOpKind>()?;
    module.add_class::<QpuStmt>()?;
    module.add_class::<ClassicalStmt>()?;
    module.add_class::<QpuFunctionDef>()?;
    module.add_class::<ClassicalFunctionDef>()?;
    module.add_class::<Backend>()?;
    module.add_class::<Program>()?;
    module.add_class::<TypeEnv>()?;
    module.add_class::<MacroEnv>()?;
    module.add_class::<DimVar>()?;
    module.add_class::<DimExpr>()?;
    module.add_class::<ExprMacroPattern>()?;
    module.add_class::<BasisMacroPattern>()?;
    module.add_class::<RecDefParam>()?;
    module.add_class::<QpuPrelude>()?;
    module.add_class::<ReplState>()?;
    module.add_class::<PlainQpuExpr>()?;
    module.add_class::<PlainQpuStmt>()?;

    Ok(())
}
