//! Wraps qwerty_ast::repl::ReplState in a Python object. Used by repl.py to
//! run the Qwerty REPL.

use crate::wrap_ast::Expr;
use qwerty_ast::ast::{Type as RustType};
use qwerty_ast::typecheck::{TypeEnv as RustTypeEnv, typecheck_expr};
use pyo3::prelude::*;
use qwerty_ast::repl;
use std::sync::Mutex;

/// Thin wrapper for qwerty_ast::repl::ReplState.
#[pyclass]
pub struct ReplState {
    // Mutex used here because PyO3 requires #[pyclass]es to be Sync, i.e.,
    // threadsafe, but ReplState is not Sync because it contains QuantumSim
    // which contains a non-Sync RefCell.
    state: Mutex<repl::ReplState>,
}

#[pymethods]
impl ReplState {
    #[new]
    fn new() -> Self {
        Self {
            state: Mutex::new(repl::ReplState::new()),
        }
    }

    fn run(&self, expr: Expr) -> Expr {
        Expr::new(self.state.lock().unwrap().run(&expr.expr))
    }

}


#[pyclass]
pub struct TypeEnv {
    check: RustTypeEnv,
}

#[pymethods]
impl TypeEnv {
    #[new]
    pub fn new() -> Self {
        TypeEnv {
            check: RustTypeEnv::new(),
        }
    }

    pub fn typecheck_expr(&mut self, expr: &Expr) -> PyResult<Type> {
        match typecheck_expr(&expr.expr, &mut self.check) {
            Ok(ty) => Ok(Type { check: ty }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("{e:?}"))),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Type {
    pub check: RustType,
}