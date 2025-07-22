//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    Adjoint, BitLiteral, Conditional, Expr, Predicated, QLit, QubitRef, Stmt, Tensor, UnitLiteral,
};
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
        if let Stmt::Expr(expr) = stmt {
            expr.eval_to_value(self)
        } else {
            Expr::UnitLiteral(UnitLiteral { dbg: None })
        }
    }
}

impl Expr {
    pub fn is_value(&self) -> bool {
        match self {
            Expr::Variable(_) => false,
            Expr::UnitLiteral(_) => true,
            Expr::Adjoint(Adjoint { func, .. }) => func.is_value(),
            Expr::Pipe(_) => false,
            Expr::Measure(_) => true,
            Expr::Discard(_) => true,
            Expr::Tensor(Tensor { vals, .. }) => vals
                .iter()
                .all(|v| v.is_value() && !matches!(v, Expr::UnitLiteral { .. })),
            Expr::BasisTranslation(_) => true,
            Expr::Predicated(Predicated {
                then_func,
                else_func,
                ..
            }) => then_func.is_value() && else_func.is_value(),
            Expr::NonUniformSuperpos(_) => false,
            Expr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => then_expr.is_value() && else_expr.is_value() && cond.is_value(),
            Expr::QLit(_) => false,
            Expr::BitLiteral(BitLiteral { dim, .. }) => *dim == 1,
            Expr::QubitRef(_) => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            Expr::QLit(qlit) => match qlit {
                QLit::ZeroQubit { .. } => Some(Expr::QubitRef(QubitRef {
                    index: state.sim.allocate(),
                })),
                _ => todo!("Rest of QLit"),
            },
            Expr::QubitRef { .. } => None,
            _ => todo!("eval_step()"),
        }
    }

    pub fn eval_to_value(&self, state: &mut ReplState) -> Expr {
        let mut expr = self.clone();
        loop {
            match expr.eval_step(state) {
                Some(new_expr) => {
                    expr = new_expr;
                }
                None => {
                    return expr;
                }
            }
        }
    }
}
