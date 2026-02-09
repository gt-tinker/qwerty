//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    Canonicalizable, Stmt, angles_are_approx_equal,
    qpu::{
        Adjoint, BitLiteral, Conditional, EmbedClassical, Expr, Predicated, QLit, QLitExpr,
        QubitRef, Tensor, UnitLiteral, expr,
    },
};
use dashu::integer::UBig;
use num_complex::Complex64;
use quantum_sparse_sim::QuantumSim;
use qwerty_ast_macros::rebuild;
use std::collections::HashMap;

/// Newtype for a `qir_runner` sparse state vector.
#[derive(Debug, Clone)]
pub struct SparseReplState {
    statevec: Vec<(UBig, Complex64)>,
    num_qubits: usize,
}

impl SparseReplState {
    /// Try to extract a one-qubit statevector for the qubit at the provided
    /// index. Returns `None` if the qubit is entangled or out of range.
    pub fn try_extract_1q_state(&self, qubit_index: usize) -> Option<(Complex64, Complex64)> {
        let SparseReplState {
            statevec,
            num_qubits,
        } = self;

        if qubit_index >= *num_qubits {
            return None;
        }

        // TODO: do stuff from notes
        None
    }
}

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

    pub fn get_sparse_state(&mut self) -> SparseReplState {
        // Convert from qir-runner bigint library (num_complex) to the one we
        // use (dashu, a fork that is supposedly faster)
        let (stefan_state, num_qubits) = self.sim.get_state();
        let statevec = stefan_state
            .into_iter()
            .map(|(bits, amplitude)| {
                let bytes_le = bits.to_bytes_le();
                let dashu_bits = UBig::from_le_bytes(&bytes_le);
                (dashu_bits, amplitude)
            })
            .collect();
        SparseReplState {
            statevec,
            num_qubits,
        }
    }
}

impl Expr {
    /// Returns a version of this expression with any q[i]s.
    pub fn recover(self, state: &SparseReplState) -> Self {
        rebuild!(Expr, self, recover, state)
    }

    pub(crate) fn recover_rewriter(self, state: &SparseReplState) -> Self {
        match self {
            Expr::QubitRef(QubitRef { index }) => {
                // TODO: actually do a partial trace etc
                Expr::QLitExpr(QLitExpr {
                    qlit: QLit::ZeroQubit { dbg: None },
                    dbg: None,
                })
            }
            other_expr => other_expr,
        }
    }

    /// Render this expression to a string in which all q[i]s replaced with
    /// equivalent qubit literals, ready to be displayed to an eager, youthful
    /// user who has no idea what q[i] means.
    pub fn render(self, state: &SparseReplState) -> String {
        let canon = self.canonicalize();
        let recovered = canon.recover(state);
        recovered.to_string()
    }

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
            Expr::QLitExpr(QLitExpr { qlit, .. }) => match qlit.clone().canonicalize() {
                QLit::ZeroQubit { .. } => Some(Expr::QubitRef(QubitRef {
                    index: state.sim.allocate(),
                })),
                QLit::OneQubit { .. } => {
                    let index = state.sim.allocate();
                    state.sim.x(index);
                    Some(Expr::QubitRef(QubitRef { index }))
                }
                QLit::UniformSuperpos { q1, q2, .. } => match (*q1, *q2) {
                    // '0' + '1'
                    (QLit::ZeroQubit { .. }, QLit::OneQubit { .. }) => {
                        let index = state.sim.allocate();
                        state.sim.h(index);
                        Some(Expr::QubitRef(QubitRef { index }))
                    }

                    // '0' - '1'
                    (QLit::ZeroQubit { .. }, QLit::QubitTilt { q, angle_deg, .. })
                        if angles_are_approx_equal(angle_deg, 180.0)
                            && matches!(*q, QLit::OneQubit { .. }) =>
                    {
                        let index = state.sim.allocate();
                        state.sim.x(index);
                        state.sim.h(index);
                        Some(Expr::QubitRef(QubitRef { index }))
                    }

                    // '1' - '0'
                    (QLit::OneQubit { .. }, QLit::QubitTilt { q, angle_deg, .. })
                        if angles_are_approx_equal(angle_deg, 180.0)
                            && matches!(*q, QLit::ZeroQubit { .. }) =>
                    {
                        let index = state.sim.allocate();
                        state.sim.x(index);
                        state.sim.h(index);
                        state.sim.x(index);
                        Some(Expr::QubitRef(QubitRef { index }))
                    }

                    // -'0' - '1'
                    (
                        QLit::QubitTilt {
                            q: q1,
                            angle_deg: angle_deg1,
                            ..
                        },
                        QLit::QubitTilt {
                            q: q2,
                            angle_deg: angle_deg2,
                            ..
                        },
                    ) if angles_are_approx_equal(angle_deg1, 180.0)
                        && angles_are_approx_equal(angle_deg2, 180.0)
                        && matches!(*q1, QLit::ZeroQubit { .. })
                        && matches!(*q2, QLit::OneQubit { .. }) =>
                    {
                        let index = state.sim.allocate();
                        state.sim.x(index);
                        state.sim.z(index);
                        state.sim.x(index);
                        state.sim.h(index);
                        Some(Expr::QubitRef(QubitRef { index }))
                    }

                    // '0' + '1'@90
                    (QLit::ZeroQubit { .. }, QLit::QubitTilt { q, angle_deg, .. })
                        if angles_are_approx_equal(angle_deg, 90.0)
                            && matches!(*q, QLit::OneQubit { .. }) =>
                    {
                        let index = state.sim.allocate();
                        state.sim.h(index);
                        state.sim.s(index);
                        Some(Expr::QubitRef(QubitRef { index }))
                    }

                    // '0' + '1'@270 ==> 'j'
                    (QLit::ZeroQubit { .. }, QLit::QubitTilt { q, angle_deg, .. })
                        if angles_are_approx_equal(angle_deg, 270.0)
                            && matches!(*q, QLit::OneQubit { .. }) =>
                    {
                        let index = state.sim.allocate();
                        state.sim.h(index);
                        state.sim.sadj(index);
                        Some(Expr::QubitRef(QubitRef { index }))
                    }

                    _ => todo!("Unknown type of superpos. Sorry"),
                },
                _ => todo!("Unknown type of qlit. Sorry"),
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
