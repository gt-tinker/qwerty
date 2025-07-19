//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{Expr, Stmt};
use quantum_sparse_sim::QuantumSim;
use std::collections::HashMap;

/// Holds the quantum simulator state and a mapping of names to values.
pub struct ReplState {
    sim: QuantumSim,
    bindings: HashMap<String, Expr>, // TODO: figure out what expr should be
}

impl ReplState {
    /// Creates a new ReplState with no qubits allocated and no names bound.
    pub fn new() -> Self {
        ReplState {
            sim: QuantumSim::new(None),
            bindings: HashMap::new(),
        }
    }

    /// Evaluates an expression and returns a value.
    pub fn run(&mut self, stmt: &Stmt) -> Expr {
        // TODO: should this return a Value (if that exists) instead of an Expr? depends on the choice above

        // TODO: run this expression

        // TODO: return the value resulting from evaluation instead of copying the input
        if let Stmt::Expr { expr, .. } = stmt {
            expr.clone()
        } else {
            Expr::UnitLiteral { dbg: None }
        }
    }
}
