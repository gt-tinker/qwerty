use pyo3::{prelude::*, types::PyType};
use qwerty_ast::{ast, dbg, typecheck};

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub struct DebugLoc {
    pub dbg: dbg::DebugLoc,
}

#[pymethods]
impl DebugLoc {
    #[new]
    fn new(file: String, line: usize, col: usize) -> Self {
        Self {
            dbg: dbg::DebugLoc { file, line, col },
        }
    }

    fn get_col(&self) -> usize {
        self.dbg.col
    }

    fn get_line(&self) -> usize {
        self.dbg.line
    }
}

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub enum RegKind {
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
pub struct Type {
    pub ty: ast::Type,
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
    fn new_reg(_cls: &Bound<'_, PyType>, elem_ty: RegKind, dim: usize) -> Self {
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
pub struct TypeEnv {
    pub env: typecheck::TypeEnv,
}

#[pymethods]
impl TypeEnv {
    #[new]
    pub fn new() -> Self {
        TypeEnv {
            env: typecheck::TypeEnv::new(),
        }
    }
}
