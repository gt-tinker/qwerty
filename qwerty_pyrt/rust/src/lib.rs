use pyo3::{prelude::*, types::PyType};
use qwerty_ast::{ast, dbg};

#[pyclass]
#[derive(Clone)]
struct DebugLoc {
    dbg: dbg::DebugLoc,
}

#[pymethods]
impl DebugLoc {
    #[new]
    fn new(file: String, line: usize, col: usize) -> Self {
        Self {
            dbg: dbg::DebugLoc { file, line, col },
        }
    }
}

#[pyclass]
#[derive(Clone)]
enum RegKind {
    Bit,
    Qubit,
    Basis,
}

// Intentionally not annotated with #[pymethods]: this is only for use in Rust
impl RegKind {
    fn to_ast_kind(&self) -> ast::RegKind {
        match self {
            RegKind::Bit => ast::RegKind::Bit,
            RegKind::Qubit => ast::RegKind::Qubit,
            RegKind::Basis => ast::RegKind::Basis,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct Type {
    ty: ast::Type,
}

#[pymethods]
impl Type {
    #[classmethod]
    fn new_func(_cls: &Bound<'_, PyType>, in_ty: Type, out_ty: Type) -> Self {
        Self {
            ty: ast::Type::FuncType {
                in_ty: Box::new(in_ty.ty),
                out_ty: Box::new(out_ty.ty),
            },
        }
    }

    #[classmethod]
    fn new_rev_func(_cls: &Bound<'_, PyType>, in_out_ty: Type) -> Self {
        Self {
            ty: ast::Type::RevFuncType {
                in_out_ty: Box::new(in_out_ty.ty),
            },
        }
    }

    #[classmethod]
    fn new_reg(_cls: &Bound<'_, PyType>, elem_ty: RegKind, dim: u64) -> Self {
        Self {
            ty: ast::Type::RegType {
                elem_ty: elem_ty.to_ast_kind(),
                dim: dim,
            },
        }
    }

    #[classmethod]
    fn new_unit(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            ty: ast::Type::UnitType,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct FunctionDef {
    function_def: ast::FunctionDef,
}

#[pymethods]
impl FunctionDef {
    #[new]
    fn new(name: String, args: Vec<(Type, String)>, ret_type: Type, dbg: Option<DebugLoc>) -> Self {
        Self {
            function_def: ast::FunctionDef {
                name,
                args: args
                    .iter()
                    .map(|(arg_ty, arg_name)| (arg_ty.ty.clone(), arg_name.to_string()))
                    .collect(),
                ret_type: ret_type.ty.clone(),
                body: vec![],
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
struct Program {
    program: ast::Program,
}

#[pymethods]
impl Program {
    #[new]
    fn new(dbg: Option<DebugLoc>) -> Self {
        Self {
            program: ast::Program {
                funcs: vec![],
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn add_function_def(&mut self, func: FunctionDef) {
        self.program.funcs.push(func.function_def);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_qwerty_pyrt")]
fn qwerty_pyrt(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let ast_submodule = PyModule::new(py, "ast")?;
    ast_submodule.add_class::<DebugLoc>()?;
    ast_submodule.add_class::<RegKind>()?;
    ast_submodule.add_class::<Type>()?;
    ast_submodule.add_class::<FunctionDef>()?;
    ast_submodule.add_class::<Program>()?;

    module.add_submodule(&ast_submodule)?;
    Ok(())
}
