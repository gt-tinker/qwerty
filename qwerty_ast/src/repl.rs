//! This module holds the state of evaluation in the Qwerty REPL and how steps
//! of evaluation are taken. The latter is based loosely on Appendix A of
//! arXiv:2404.12603.

use crate::ast::{
    Assign, Canonicalizable, Stmt, StmtExpr, angle_is_approx_zero, angles_are_approx_equal,
    canon_angle,
    qpu::{
        Adjoint, Basis, BasisTranslation, BitLiteral, Expr, Measure, NonUniformSuperpos, Pipe,
        Predicated, QLit, QLitExpr, QubitRef, Tensor, UnitLiteral, Variable, expr,
    },
};
use dashu::{base::BitTest, integer::UBig};
use num_bigint::BigUint;
use num_complex::{Complex64, ComplexFloat};
use quantum_sparse_sim::QuantumSim;
use qwerty_ast_macros::rebuild;
use std::{collections::HashMap, fmt};

/// Newtype for a `qir_runner` sparse state vector.
#[derive(Debug, Clone)]
pub struct SparseReplState {
    statevec: Vec<(UBig, Complex64)>,
    num_qubits: usize,
}

impl fmt::Display for SparseReplState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let SparseReplState {
            statevec,
            num_qubits,
        } = self;

        write!(f, "[")?;

        for (i, (bits, amp)) in statevec.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }

            write!(f, "'")?;

            for i in 0..*num_qubits {
                write!(
                    f,
                    "{}",
                    if bits.bit(num_qubits - 1 - i) {
                        "1"
                    } else {
                        "0"
                    }
                )?;
            }

            write!(f, "': {}", amp)?;
        }

        write!(f, "]")
    }
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

        // Fast path: this is a one-qubit statevector already
        if *num_qubits == 1 {
            return match &statevec[..] {
                [(bits, amp0)] if *bits == UBig::ZERO => Some((*amp0, Complex64::ZERO)),
                [(bits, amp1)] if *bits == UBig::ONE => Some((Complex64::ZERO, *amp1)),
                [(bits0, amp0), (bits1, amp1)] | [(bits1, amp1), (bits0, amp0)]
                    if *bits0 == UBig::ZERO && *bits1 == UBig::ONE =>
                {
                    Some((*amp0, *amp1))
                }
                _ => unreachable!("Not a one-qubit unit vector"),
            };
        }

        // Let ρ = |ψ⟩⟨ψ|, where |ψ⟩ is the entire n-qubit statevector, i.e.,
        // |ψ⟩ = ∑_{i=0}^{2ⁿ−1} αᵢ.
        //
        // Then let ρₖ= tr₋ₖ(ρ). Here, tr₋ₖ(⋅) is notation for the partial
        // trace as defined in Section 2.4.3 of Mike&Ike, with the ₋ₖ subscript
        // meaning "trace out all qubits except for the qubit with index k".
        //
        // Our first step is to check if tr(ρₖ²) = 1, that is, if the qubit at
        // index k is not entangled with any other qubits. Per some
        // pen-and-paper perspiration,
        // tr(ρₖ²) = ∑_{ℓ=0}^{2ⁿ⁻¹−1} ∑_{ℓ'=0}^{2ⁿ⁻¹−1}
        //           ∑_{x=0}^1 ∑_{y=0}^1
        //           α_{ℓxℓ} α_{ℓyℓ}* α_{ℓ'yℓ'} α_{ℓ'xℓ'}*
        //         = ∑_{ℓ=0}^{2ⁿ⁻¹−1} ∑_{ℓ'=0}^{2ⁿ⁻¹−1}
        //           α_{ℓ0ℓ} α_{ℓ0ℓ}* α_{ℓ'0ℓ'} α_{ℓ'0ℓ'}*
        //           + α_{ℓ0ℓ} α_{ℓ1ℓ}* α_{ℓ'1ℓ'} α_{ℓ'0ℓ'}*
        //           + α_{ℓ1ℓ} α_{ℓ0ℓ}* α_{ℓ'0ℓ'} α_{ℓ'1ℓ'}*
        //           + α_{ℓ1ℓ} α_{ℓ1ℓ}* α_{ℓ'1ℓ'} α_{ℓ'1ℓ'}*
        //         = ∑_{ℓ=0}^{2ⁿ⁻¹−1} ∑_{ℓ'=0}^{2ⁿ⁻¹−1}
        //           |α_{ℓ0ℓ}|² |α_{ℓ'0ℓ'}|²
        //           + α_{ℓ0ℓ} α_{ℓ1ℓ}* α_{ℓ'1ℓ'} α_{ℓ'0ℓ'}*
        //           + (α_{ℓ0ℓ} α_{ℓ1ℓ}* α_{ℓ'1ℓ'} α_{ℓ'0ℓ'}*)*
        //           + |α_{ℓ1ℓ}|² |α_{ℓ'1ℓ'}|²
        //         = ∑_{ℓ=0}^{2ⁿ⁻¹−1} ∑_{ℓ'=0}^{2ⁿ⁻¹−1}
        //           |α_{ℓ0ℓ}|² |α_{ℓ'0ℓ'}|²
        //           + 2Re[α_{ℓ0ℓ} α_{ℓ1ℓ}* α_{ℓ'1ℓ'} α_{ℓ'0ℓ'}*]
        //           + |α_{ℓ1ℓ}|² |α_{ℓ'1ℓ'}|²
        //
        // Beware, * here means complex conjugate, not multiplication. And the
        // shitty ℓxℓ notation means bits [0,k) of ℓ, concatenated with x,
        // concatenated with bits [k,n) of ℓ. Sorry.

        // First, move the bit in question to the LSB (rightmost) position and
        // sort the vector. Amplitudes whose bits differe only by the bit in
        // question are adjacent in the sorted vector.
        let bit_idx = *num_qubits - 1 - qubit_index;
        let mut statevec: Vec<_> = statevec
            .iter()
            .map(|(bits, amp)| {
                let split_at = bit_idx + 1;
                let (mut lo, hi) = bits.clone().split_bits(split_at);
                let bit = lo.bit(bit_idx);
                lo.clear_bit(bit_idx);
                lo <<= 1;
                if bit {
                    lo.set_bit(0);
                }
                let bits = hi << split_at | lo;
                (bits, *amp)
            })
            .collect();
        statevec.sort_by(|(lbits, _lamp), (rbits, _ramp)| lbits.cmp(rbits));

        // Next, find pairs of adjacent amplitudes. `pairs` will contain pairs
        // of nonzero amplitudes that differ only by the LSB. The first entry
        // of the pair has a 0 LSB, and the second has a 1 LSB.
        let mut pairs = Vec::new();
        let mut i = 0;
        loop {
            if i + 1 >= statevec.len() {
                break;
            }
            let (bits0, amp0) = &statevec[i];
            let (bits1, amp1) = &statevec[i + 1];
            if bits0 ^ bits1 == UBig::ONE {
                pairs.push((*amp0, *amp1));
                // Skip past r
                i += 1;
            }

            i += 1;
        }

        // Calculate tr(ρₖ²) to find out if this qubit is entangled.
        let mut trace = 0.0;

        // Calculate the first and last terms in the sum above.
        for (bits, amp) in statevec.iter() {
            let bit = bits.bit(0);
            for (bitsp, ampp) in statevec.iter() {
                let bitp = bitsp.bit(0);
                if bit == bitp {
                    trace += amp.norm_sqr() * ampp.norm_sqr();
                }
            }
        }

        // Calculate the middle term in the sum above.
        for (amp0, amp1) in pairs.iter() {
            for (amp0p, amp1p) in pairs.iter() {
                trace += 2.0 * (amp0 * amp1.conj() * amp1p * amp0p.conj()).re();
            }
        }

        if !angles_are_approx_equal(trace, 1.0) {
            // Entangled!
            return None;
        }
        // Else, not entangled. There is hope.

        // Next, we want to find a |ψₖ⟩ such that ρₖ = |ψₖ⟩⟨ψₖ|. By some
        // pen-and-paper math, we have
        // ρₖ= ∑_{x=0}^1 ∑_{y=0}^1 β_{xy}|x⟩⟨y|,
        // where β_{xy} = ∑_{ℓ=0}^{2ⁿ⁻¹−1} α_{ℓxℓ} α_{ℓyℓ}*.
        //
        // To find |ψₖ⟩, we can do one of the two following computations:
        //
        //    ρₖ|0⟩      √(∑_{ℓ=0}^{2ⁿ⁻¹−1} |α_{ℓ0ℓ}|²) |0⟩ + ∑_{ℓ=0}^{2ⁿ⁻¹−1} α_{ℓ1ℓ} α_{ℓ0ℓ}* |1⟩
        // ----------- =                                      ---------------------------------
        //  √(⟨0|ρₖ|0⟩)                                         √(∑_{ℓ=0}^{2ⁿ⁻¹−1} |α_{ℓ0ℓ}|²)
        //
        // or
        //
        //    ρₖ|1⟩      ∑_{ℓ=0}^{2ⁿ⁻¹−1} α_{ℓ0ℓ} α_{ℓ1ℓ}* |0⟩ + √(∑_{ℓ=0}^{2ⁿ⁻¹−1} |α_{ℓ1ℓ}|²) |1⟩
        // ----------- = ---------------------------------
        //  √(⟨1|ρₖ|1⟩)    √(∑_{ℓ=0}^{2ⁿ⁻¹−1} |α_{ℓ1ℓ}|²)
        //
        // The right one to choose depends on which (if any) of the
        // denominators √(⟨0|ρₖ|0⟩) and √(⟨1|ρₖ|1⟩) are 0.

        let mut beta00 = 0.0;
        let mut beta11 = 0.0;

        for (bits, amp) in statevec {
            if !bits.bit(0) {
                // LSB is 0
                beta00 += amp.norm_sqr();
            } else {
                // LSB is 1
                beta11 += amp.norm_sqr();
            }
        }

        // Decide which way to use to compute |ψₖ⟩.
        if !angle_is_approx_zero(beta00) {
            // Compute |ψₖ⟩ = ρₖ|0⟩/√(⟨0|ρₖ|0⟩).
            let beta00hat = beta00.sqrt();
            let mut beta10 = Complex64::ZERO;
            for (amp0, amp1) in pairs {
                beta10 += amp1 * amp0.conj();
            }
            let beta10hat = beta10 / beta00hat;
            Some((beta00hat.into(), beta10hat))
        } else {
            // Compute |ψₖ⟩ = ρₖ|1⟩/√(⟨1|ρₖ|1⟩).
            let beta11hat = beta11.sqrt();
            let mut beta01 = Complex64::ZERO;
            for (amp0, amp1) in pairs {
                beta01 += amp0 * amp1.conj();
            }
            let beta01hat = beta01 / beta11hat;
            Some((beta01hat, beta11hat.into()))
        }
    }
}

/// Holds the quantum simulator state and a mapping of names to values.
pub struct ReplState {
    sim: QuantumSim,
    bindings: HashMap<String, Expr>,
}

/// Swap both endianness and the bigint library used. Stefan (qir-runner)
/// puts bit 0 on the rightmost bit (the LSB), but for us, bit 0 is the
/// leftmost (MSB).
fn stefan_bits_to_our_bits(stefan_bits: BigUint, num_bits: usize) -> UBig {
    let mut our_bits = UBig::ZERO;
    for bit_idx in 0..num_bits {
        let bit = if stefan_bits.bit(bit_idx as u64) {
            UBig::ONE
        } else {
            UBig::ZERO
        };
        our_bits = our_bits << 1 | bit;
    }
    our_bits
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
    pub fn run(&mut self, stmt: &Stmt<Expr>) -> Expr {
        match stmt {
            Stmt::Expr(StmtExpr { expr, .. }) => expr.eval_to_value(self),
            Stmt::Assign(Assign { lhs, rhs, .. }) => {
                let rhs_val = rhs.eval_to_value(self);

                if let Some(old_rhs_val) = self.bindings.insert(lhs.clone(), rhs_val) {
                    self.free_value(&old_rhs_val);
                }

                Expr::UnitLiteral(UnitLiteral { dbg: None })
            }
            unknown => todo!("Unknown type of statment {:?}. Sorry", unknown),
        }
    }

    /// Discards this value, freeing any resources associated with it.
    /// Realistically, this means discarding qubit references.
    pub fn free_value(&mut self, expr: &Expr) {
        // Per the grammar, these are the only ways that `q[i]`s can show up in
        // a value.
        match expr {
            Expr::QubitRef(QubitRef { index }) => {
                self.sim.release(*index);
            }
            Expr::Tensor(Tensor { vals, .. }) => {
                for val in vals {
                    self.free_value(val);
                }
            }
            _ => {}
        }
    }

    pub fn get_sparse_state(&mut self) -> SparseReplState {
        // Convert from qir-runner bigint library (num_bigint) to the one we
        // use (dashu, a fork that is supposedly faster)
        let (stefan_state, num_qubits) = self.sim.get_state();
        let statevec = stefan_state
            .into_iter()
            .map(|(stefan_bits, amplitude)| {
                let our_bits = stefan_bits_to_our_bits(stefan_bits, num_qubits);
                (our_bits, amplitude)
            })
            .collect();
        SparseReplState {
            statevec,
            num_qubits,
        }
    }
}

impl QubitRef {
    /// Try to recover an expression, e.g., `.25*'0' + .75*'1'` from this
    /// `q[i]` given the repl state.
    fn recover(self, state: &SparseReplState) -> Expr {
        let QubitRef { index } = self;

        if let Some((amp0, amp1)) = state.try_extract_1q_state(index) {
            let (zero_abs, zero_rad) = amp0.to_polar();
            let zero_prob = zero_abs * zero_abs;
            let zero_deg = canon_angle(zero_rad * std::f64::consts::FRAC_1_PI * 180.0);

            let (one_abs, one_rad) = amp1.to_polar();
            let one_prob = one_abs * one_abs;
            let one_deg = canon_angle(one_rad * std::f64::consts::FRAC_1_PI * 180.0);

            let zero = QLit::ZeroQubit { dbg: None };
            let zero_vec = if angle_is_approx_zero(zero_deg) {
                zero
            } else {
                QLit::QubitTilt {
                    q: Box::new(zero),
                    angle_deg: zero_deg,
                    dbg: None,
                }
            };

            let one = QLit::OneQubit { dbg: None };
            let one_vec = if angle_is_approx_zero(one_deg) {
                one
            } else {
                QLit::QubitTilt {
                    q: Box::new(one),
                    angle_deg: one_deg,
                    dbg: None,
                }
            };

            if angle_is_approx_zero(zero_prob) {
                Expr::QLitExpr(QLitExpr {
                    qlit: one_vec,
                    dbg: None,
                })
            } else if angle_is_approx_zero(one_prob) {
                Expr::QLitExpr(QLitExpr {
                    qlit: zero_vec,
                    dbg: None,
                })
            } else if angles_are_approx_equal(zero_prob, 0.5) {
                let uniform = QLit::UniformSuperpos {
                    q1: Box::new(zero_vec),
                    q2: Box::new(one_vec),
                    dbg: None,
                };
                Expr::QLitExpr(QLitExpr {
                    qlit: uniform,
                    dbg: None,
                })
            } else {
                Expr::NonUniformSuperpos(NonUniformSuperpos {
                    pairs: vec![(zero_prob, zero_vec), (one_prob, one_vec)],
                    dbg: None,
                })
            }
        } else {
            // Pass through unchanged
            Expr::QubitRef(QubitRef { index })
        }
    }
}

impl QLit {
    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self.clone().canonicalize() {
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
        }
    }
}

impl Expr {
    /// Returns a version of this expression with any `q[i]`s replaced with
    /// equivalent qubit literals.
    pub fn recover(self, state: &SparseReplState) -> Self {
        rebuild!(Expr, self, recover, state)
    }

    pub(crate) fn recover_rewriter(self, state: &SparseReplState) -> Self {
        match self {
            Expr::QubitRef(qref) => qref.recover(state),
            other_expr => other_expr,
        }
    }

    /// Render this expression to a string in which all q[i]s replaced with
    /// equivalent qubit literals, ready to be displayed to an eager, youthful
    /// user who has no idea what q[i] means. Returns an empty string if
    /// nothing should be printed.
    pub fn render(self, state: &SparseReplState) -> String {
        if let Expr::UnitLiteral(_) = self {
            // In the Python REPL, nothing is printed for None.
            "".to_string()
        } else {
            let canon = self.canonicalize();
            let recovered = canon.recover(state);
            recovered.to_string()
        }
    }

    pub fn is_value(&self) -> bool {
        match self {
            Expr::Variable(_) => false,
            Expr::UnitLiteral(_) => true,
            Expr::EmbedClassical(_) => true,
            Expr::Adjoint(Adjoint { func, .. }) => func.is_value(),
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
            }) => then_func.is_value() && else_func.is_value(),
            Expr::NonUniformSuperpos(_) => false,
            Expr::Ensemble(_) => false,
            Expr::Conditional(_) => false,
            Expr::QLitExpr(_) => false,
            Expr::BitLiteral(BitLiteral { n_bits, .. }) => *n_bits == 1,
            Expr::QubitRef(_) => true,
        }
    }

    pub fn eval_step(&self, state: &mut ReplState) -> Option<Expr> {
        match self {
            // Normally, we would have something like E-Let* in Figure 11-4 of
            // TAPL. But we are in a weird situation where we re getting a
            // trickle of statements.
            Expr::Variable(Variable { name, .. }) => match state.bindings.get(name) {
                None => unreachable!("Unbound variable {name}. Type checking should catch this!"),
                Some(rhs_val) => Some(rhs_val.clone()),
            },

            Expr::Pipe(Pipe { lhs, rhs, .. }) => {
                match (&**lhs, &**rhs) {
                    // E-Meas
                    (Expr::QubitRef(QubitRef { index }), Expr::Measure(Measure { basis, .. })) => {
                        let basis = basis.clone().canonicalize();
                        if basis.is_std_1q() {
                            let outcome = state.sim.measure(*index);
                            state.sim.release(*index);

                            let val = if !outcome { UBig::ZERO } else { UBig::ONE };
                            Some(Expr::BitLiteral(BitLiteral {
                                val,
                                n_bits: 1,
                                dbg: None,
                            }))
                        } else {
                            let std1 = Basis::std(1, None);
                            Some(Expr::Pipe(Pipe {
                                lhs: Box::new(Expr::Pipe(Pipe {
                                    lhs: Box::new(Expr::QubitRef(QubitRef { index: *index })),
                                    rhs: Box::new(Expr::BasisTranslation(BasisTranslation {
                                        bin: basis.clone(),
                                        bout: std1.clone(),
                                        dbg: None,
                                    })),
                                    dbg: None,
                                })),
                                rhs: Box::new(Expr::Measure(Measure {
                                    basis: std1,
                                    dbg: None,
                                })),
                                dbg: None,
                            }))
                        }
                    }

                    // E-Pipe2
                    (lhs, rhs_val) if rhs_val.is_value() => {
                        let lhs = lhs.eval_step(state)?;
                        Some(Expr::Pipe(Pipe {
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs_val.clone()),
                            dbg: None,
                        }))
                    }

                    // E-Pipe1
                    (lhs, rhs) => {
                        let rhs = rhs.eval_step(state)?;
                        Some(Expr::Pipe(Pipe {
                            lhs: Box::new(lhs.clone()),
                            rhs: Box::new(rhs),
                            dbg: None,
                        }))
                    }
                }
            }

            // E-QAtom
            Expr::QLitExpr(QLitExpr { qlit, .. }) => qlit.eval_step(state),

            // By definition, values do not need evaluation
            val if val.is_value() => None,

            unknown => todo!("eval_step() for Expr: {:?}", unknown),
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
