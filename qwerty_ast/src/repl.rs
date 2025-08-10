//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    BitLiteral, Stmt,
    qpu::{
        Adjoint, Conditional, EmbedClassical, Expr, Predicated, QLit, QubitRef, Tensor, UnitLiteral,
    },
};
use quantum_sparse_sim::QuantumSim;
use std::collections::HashMap;

/// Holds the quantum simulator state and a mapping of names to values.
pub struct ReplState {
    sim: QuantumSim,
    // TODO: use this
    _bindings: HashMap<String, Expr>,
}

impl ReplState {
    /// Creates a new ReplState with no qubits allocated and no names bound.
    pub fn new() -> Self {
        ReplState {
            sim: QuantumSim::new(None),
            _bindings: HashMap::new(),
        }
    }

    /// Evaluates an expression and returns a value.
    pub fn run(&mut self, stmt: &Stmt<Expr>) -> Expr {
        if let Stmt::Expr(stmt_expr) = stmt {
            stmt_expr.expr.eval_to_value(self)
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
            Expr::EmbedClassical(EmbedClassical { func_name: _, .. }) => true,
            Expr::Adjoint(Adjoint { func, .. }) => func.as_ref().is_value(),
            Expr::Pipe(_) => false,
            Expr::Measure(_) => true,
            Expr::Discard(_) => true,
            Expr::Tensor(Tensor { vals, .. }) => vals
                .iter()
                .all(|v| v.is_value() && !matches!(v, Expr::UnitLiteral(_))),
            Expr::BasisTranslation(_) => true,
            Expr::Predicated(Predicated {
                then_func,
                else_func,
                ..
            }) => then_func.as_ref().is_value() && else_func.as_ref().is_value(),
            Expr::NonUniformSuperpos(_) => false,
            Expr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => {
                then_expr.as_ref().is_value()
                    && else_expr.as_ref().is_value()
                    && cond.as_ref().is_value()
            }
            Expr::QLit(_) => false,
            Expr::BitLiteral(BitLiteral { n_bits, .. }) => *n_bits == 1,
            Expr::QubitRef(_) => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            Expr::QLit(qlit) => match qlit {
                QLit::ZeroQubit { .. } => Some(Expr::QubitRef(QubitRef {
                    index: state.sim.allocate(),
                })),
                _ => todo!("Rest of QLit eval_step"),
            },
            Expr::QubitRef { .. } => None,
            _ => todo!("eval_step() for Expr"),
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
