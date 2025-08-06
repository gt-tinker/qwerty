//! Wraps qwerty_ast::repl::ReplState in a Python object. Used by repl.py to
//! run the Qwerty REPL.

use crate::wrap_ast::{QpuExpr, QpuStmt};
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

    fn run(&self, stmt: QpuStmt) -> QpuExpr {
        QpuExpr::new(self.state.lock().unwrap().run(&stmt.stmt))
    }
}
