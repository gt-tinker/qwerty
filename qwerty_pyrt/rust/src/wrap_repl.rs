//! Wraps qwerty_ast::repl::ReplState in a Python object. Used by repl.py to
//! run the Qwerty REPL.

use crate::wrap_ast::{
    PlainClassicalFunctionDef, PlainQpuExpr, PlainQpuStmt, ProgErrKind, get_err,
};
use pyo3::prelude::*;
use qwerty_ast::repl;
use std::fmt;
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
    pub fn new() -> Self {
        Self {
            state: Mutex::new(repl::ReplState::new()),
        }
    }

    pub fn run(&mut self, py: Python<'_>, stmt: PlainQpuStmt) -> PyResult<PlainQpuExpr> {
        let expr = self.state.lock().unwrap().run(&stmt.stmt).map_err(
            |repl::NotImplementedError(msg)| get_err(py, ProgErrKind::Internal, msg, None),
        )?;
        Ok(PlainQpuExpr { expr })
    }

    pub fn free_value(&mut self, val: PlainQpuExpr) {
        self.state.lock().unwrap().free_value(&val.expr);
    }

    pub fn get_sparse_state(&mut self) -> SparseReplState {
        let state = self.state.lock().unwrap().get_sparse_state();
        SparseReplState { state }
    }

    pub fn insert_classical_func(&mut self, func_def: PlainClassicalFunctionDef) {
        self.state
            .lock()
            .unwrap()
            .insert_classical_func(func_def.function_def);
    }
}

/// Thin wrapper for qwerty_ast::repl::SparseReplState.
#[pyclass(str)]
pub struct SparseReplState {
    pub state: repl::SparseReplState,
}

impl fmt::Display for SparseReplState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.state)
    }
}
