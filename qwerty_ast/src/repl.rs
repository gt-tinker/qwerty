//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    Stmt,
    qpu::{
        Adjoint, BitLiteral, Conditional, EmbedClassical, Expr, Predicated, QLit, QLitExpr,
        QubitRef, Tensor, UnitLiteral,
    },
};
use quantum_sparse_sim::QuantumSim;
use std::collections::HashMap;
mod qlit2sparse;

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
            Expr::Ensemble(_) => false,
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
            Expr::QLitExpr(_) => false,
            Expr::BitLiteral(BitLiteral { n_bits, .. }) => *n_bits == 1,
            Expr::QubitRef(_) => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            Expr::QLitExpr(QLitExpr { qlit, .. }) => match qlit {
                QLit::ZeroQubit { .. } => Some(Expr::QubitRef(QubitRef {
                    index: state.sim.allocate(),
                })),
                _ => todo!("Rest of QLit eval_step"),

                    QLit::OneQubit { .. } => {
                        let q = state.sim.allocate();
                        state.sim.x(q); // use x to flip 0 to 1
                        Some(Expr::QubitRef(QubitRef { index: q }))
                    }
                    QLit::QubitTilt { q, angle_deg, dbg } => {
                        let inside_expr = Expr::QLitExpr(QLitExpr { qlit: *q.clone(), dbg: dbg.clone() }); // recursion for nest
                        if let Some(Expr::QubitRef(QubitRef { index })) = inside_expr.eval_step(state) {
                            state.sim.rz(*angle_deg, index); // tilt using spacesim
                            Some(Expr::QubitRef(QubitRef { index }))
                        } else {
                            None // evaluation failed
                        }
                    }
                    QLit::UniformSuperpos { q1, q2, dbg } => {
                        let inside_expr = Expr::QLitExpr(QLitExpr { qlit: *q1.clone(), dbg: dbg.clone() });
                        if let Some(Expr::QubitRef(QubitRef { index })) = inside_expr.eval_step(state) {
                            state.sim.h(index);
                            Some(Expr::QubitRef(QubitRef { index }))
                        } else {
                            None
                        }
                    }
                    // qs is a vector, so parse through vector and then evaluate
                    QLit::QubitTensor { qs, dbg } => {
                        let mut vals = Vec::new();
                        for qlit in qs {
                            let inner_expr = Expr::QLitExpr(QLitExpr { qlit: qlit.clone(), dbg: dbg.clone() });
                            if let Some(Expr::QubitRef(QubitRef { index })) = inner_expr.eval_step(state) {
                                vals.push(Expr::QubitRef(QubitRef { index }));
                            } else {
                                return None;
                            }
                        }
                        Some(Expr::Tensor(Tensor {
                            vals,
                            dbg: dbg.clone(),
                        }))
                    }
                    QLit::QubitUnit { dbg, .. } => Some(Expr::UnitLiteral(UnitLiteral { dbg: dbg.clone() })),
            },
            Expr::QubitRef { .. } | Expr::UnitLiteral { .. } => None,
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
