/*
 * QWERTY Programming Language Compiler
 * Abstract Syntax Tree (AST) Definitions
 *
 * This module defines the Abstract Syntax Tree (AST) structures
 * used for parsing and representing QWERTY programs.
 *
 * Version: 1.0
 */

use crate::dbg::DebugLoc;

// ----- Types -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    FuncType { in_ty: Box<Type>, out_ty: Box<Type> },
    RevFuncType { in_out_ty: Box<Type> },
    RegType { elem_ty: RegKind, dim: u64 },
    UnitType,
}

// ----- Registers -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegKind {
    Bit,   // Classical bit register
    Qubit, // Quantum bit register
    Basis, // Register for basis states
}

// ----- Qubit Literals -----

#[derive(Debug, Clone, PartialEq)]
pub enum QLit {
    ZeroQubit {
        dbg: Option<DebugLoc>,
    },
    OneQubit {
        dbg: Option<DebugLoc>,
    },
    QubitTilt {
        q: Box<QLit>,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    },
    UniformSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        dbg: Option<DebugLoc>,
    },
    QubitTensor {
        qs: Vec<QLit>,
        dbg: Option<DebugLoc>,
    },
}

impl QLit {
    /// Converts a qubit literal to a basis vector since in Appendix A, every ql is
    /// a bv.
    pub fn convert_to_basis_vector(&self) -> Vector {
        match self {
            QLit::ZeroQubit { dbg } => Vector::ZeroVector { dbg: dbg.clone() },
            QLit::OneQubit { dbg } => Vector::OneVector { dbg: dbg.clone() },
            QLit::QubitTilt { q, angle_deg, dbg } => Vector::VectorTilt {
                q: Box::new(q.convert_to_basis_vector()),
                angle_deg: *angle_deg,
                dbg: dbg.clone(),
            },
            QLit::UniformSuperpos { q1, q2, dbg } => Vector::UniformVectorSuperpos {
                q1: Box::new(q1.convert_to_basis_vector()),
                q2: Box::new(q2.convert_to_basis_vector()),
                dbg: dbg.clone(),
            },
            QLit::QubitTensor { qs, dbg } => Vector::VectorTensor {
                qs: qs.iter().map(QLit::convert_to_basis_vector).collect(),
                dbg: dbg.clone(),
            },
        }
    }
}

// ----- Vector -----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAtomKind {
    PadAtom,    // '?'
    TargetAtom, // '_'
}

#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
    ZeroVector {
        dbg: Option<DebugLoc>,
    },
    OneVector {
        dbg: Option<DebugLoc>,
    },
    PadVector {
        dbg: Option<DebugLoc>,
    },
    TargetVector {
        dbg: Option<DebugLoc>,
    },
    VectorTilt {
        q: Box<Vector>,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    },
    UniformVectorSuperpos {
        q1: Box<Vector>,
        q2: Box<Vector>,
        dbg: Option<DebugLoc>,
    },
    VectorTensor {
        qs: Vec<Vector>,
        dbg: Option<DebugLoc>,
    },
    VectorUnit {
        dbg: Option<DebugLoc>,
    },
}

impl Vector {
    /// Represents a vector in a human-readable form for error messages sent
    /// back to the programmer.
    pub fn to_programmer_str(&self) -> String {
        match self {
            Vector::ZeroVector { .. } => "'0'".to_string(),
            Vector::OneVector { .. } => "'1'".to_string(),
            Vector::PadVector { .. } => "'?'".to_string(),
            Vector::TargetVector { .. } => "'_'".to_string(),
            Vector::VectorTilt { q, angle_deg, .. } => {
                if off_phase(0.0, *angle_deg) {
                    format!("-{}", q.to_programmer_str())
                } else {
                    format!("({}@{})", q.to_programmer_str(), angle_deg)
                }
            }
            Vector::UniformVectorSuperpos { q1, q2, .. } => {
                format!("({} + {})", q1.to_programmer_str(), q2.to_programmer_str())
            }
            Vector::VectorTensor { qs, .. } => format!(
                "({})",
                qs.iter()
                    .map(|q| q.to_programmer_str())
                    .collect::<Vec<String>>()
                    .join(" * ")
            ),
            Vector::VectorUnit { .. } => "[]".to_string(),
        }
    }

    /// Returns number of non-target and non-padding qubits represented by a basis
    /// vector (⌊bv⌋ in the Appendix) or None if the basis vector is malformed
    /// (currently, if both sides of a superposition have different dimensions
    ///  or if a tensor product has less than two vectors).
    pub fn get_explicit_dim(&self) -> Option<usize> {
        match self {
            Vector::ZeroVector { .. } | Vector::OneVector { .. } => Some(1),
            Vector::PadVector { .. } | Vector::TargetVector { .. } => Some(0),
            Vector::VectorTilt { q: inner_bv, .. } => inner_bv.get_explicit_dim(),
            Vector::UniformVectorSuperpos {
                q1: inner_bv_1,
                q2: inner_bv_2,
                ..
            } => match (inner_bv_1.get_explicit_dim(), inner_bv_2.get_explicit_dim()) {
                (Some(inner_dim1), Some(inner_dim2)) if inner_dim1 == inner_dim2 => {
                    Some(inner_dim1)
                }
                _ => None,
            },
            Vector::VectorTensor { qs: inner_bvs, .. } => {
                if inner_bvs.len() < 2 {
                    None
                } else {
                    inner_bvs
                        .iter()
                        .try_fold(0, |acc, v| v.get_explicit_dim().map(|dim| acc + dim))
                }
            }
            Vector::VectorUnit { .. } => Some(0),
        }
    }

    /// Returns number of qubits represented by a basis vector, including
    /// padding and target qubits. (This is |bv| in the Appendix.) Returns None
    /// if the basis vector is malformed (currently, if both sides of a
    /// superposition have different dimensions or if a tensor product has less
    /// than two vectors).
    pub fn get_dim(&self) -> Option<usize> {
        match self {
            Vector::ZeroVector { .. }
            | Vector::OneVector { .. }
            | Vector::PadVector { .. }
            | Vector::TargetVector { .. } => Some(1),
            Vector::VectorTilt { q: inner_bv, .. } => inner_bv.get_dim(),
            Vector::UniformVectorSuperpos {
                q1: inner_bv_1,
                q2: inner_bv_2,
                ..
            } => match (inner_bv_1.get_dim(), inner_bv_2.get_dim()) {
                (Some(inner_dim1), Some(inner_dim2)) if inner_dim1 == inner_dim2 => {
                    Some(inner_dim1)
                }
                _ => None,
            },
            Vector::VectorTensor { qs: inner_bvs, .. } => {
                if inner_bvs.len() < 2 {
                    None
                } else {
                    inner_bvs
                        .iter()
                        .try_fold(0, |acc, v| v.get_dim().map(|dim| acc + dim))
                }
            }
            Vector::VectorUnit { .. } => Some(0),
        }
    }

    /// Returns the zero-indexed qubit positions that are correspond to the
    /// provided atom. This Ξva[bv] in Appendix A. Returning None indicates a
    /// malformed vector (currently, a superposition whose possibilities do not
    /// have matching indices or an empty tensor product).
    pub fn get_atom_indices(&self, atom: VectorAtomKind) -> Option<Vec<usize>> {
        match (self, atom) {
            (Vector::ZeroVector { .. }, _)
            | (Vector::OneVector { .. }, _)
            | (Vector::VectorUnit { .. }, _)
            | (Vector::PadVector { .. }, VectorAtomKind::TargetAtom)
            | (Vector::TargetVector { .. }, VectorAtomKind::PadAtom) => Some(vec![]),

            (Vector::PadVector { .. }, VectorAtomKind::PadAtom)
            | (Vector::TargetVector { .. }, VectorAtomKind::TargetAtom) => Some(vec![0]),

            (Vector::VectorTilt { q, .. }, _) => q.get_atom_indices(atom),

            (Vector::UniformVectorSuperpos { q1, q2, .. }, _) => {
                let left_indices = q1.get_atom_indices(atom)?;
                let right_indices = q2.get_atom_indices(atom)?;
                if left_indices == right_indices {
                    Some(left_indices)
                } else {
                    None
                }
            }

            (Vector::VectorTensor { ref qs, .. }, _) => {
                if qs.len() < 2 {
                    None
                } else {
                    let mut offset = 0;
                    let mut indices = vec![];
                    for vec in qs {
                        let vec_indices = vec.get_atom_indices(atom)?;
                        for idx in vec_indices {
                            indices.push(idx + offset);
                        }
                        let vec_dim = vec.get_dim()?;
                        offset += vec_dim;
                    }
                    Some(indices)
                }
            }
        }
    }

    /// Returns a version of this vector without any '?' or '_' atoms. Assumes
    /// the original vector is well-typed.
    pub fn make_explicit(&self) -> Vector {
        match self {
            Vector::ZeroVector { .. } | Vector::OneVector { .. } | Vector::VectorUnit { .. } => {
                self.clone()
            }
            Vector::PadVector { ref dbg } | Vector::TargetVector { ref dbg } => {
                Vector::VectorUnit { dbg: dbg.clone() }
            }
            Vector::VectorTilt {
                ref q,
                angle_deg,
                ref dbg,
            } => {
                let q_explicit = q.make_explicit();
                if let Vector::VectorUnit { .. } = q_explicit {
                    q_explicit
                } else {
                    Vector::VectorTilt {
                        q: Box::new(q_explicit),
                        angle_deg: *angle_deg,
                        dbg: dbg.clone(),
                    }
                }
            }
            Vector::UniformVectorSuperpos {
                ref q1,
                ref q2,
                ref dbg,
            } => Vector::UniformVectorSuperpos {
                q1: Box::new(q1.make_explicit()),
                q2: Box::new(q2.make_explicit()),
                dbg: dbg.clone(),
            },
            Vector::VectorTensor { ref qs, ref dbg } => {
                let qs_explicit: Vec<_> = qs
                    .iter()
                    .map(Vector::make_explicit)
                    .filter(|vec| !matches!(vec, Vector::VectorUnit { .. }))
                    .collect();
                // Make an assumption that this is well-formed
                if qs_explicit.is_empty() {
                    Vector::VectorUnit { dbg: dbg.clone() }
                } else if qs_explicit.len() == 1 {
                    qs_explicit[0].clone()
                } else {
                    Vector::VectorTensor {
                        qs: qs_explicit,
                        dbg: dbg.clone(),
                    }
                }
            }
        }
    }

    /// Returns an equivalent vector such that:
    /// 1. No tensor product contains a ()
    /// 2. No tensor product contains another tensor product
    /// 3. All @s are propagated to the outside
    /// 4. All tensor product angles in canon form (i.e., in the interval
    ///    [0.0, 360.0))
    /// Assumes this vector is well-typed.
    pub fn canonicalize(&self) -> Vector {
        match self {
            Vector::ZeroVector { .. }
            | Vector::OneVector { .. }
            | Vector::PadVector { .. }
            | Vector::TargetVector { .. }
            | Vector::VectorUnit { .. } => self.clone(),
            Vector::VectorTilt {
                ref q,
                angle_deg,
                ref dbg,
            } => {
                let q_canon = q.canonicalize();
                if let Vector::VectorTilt {
                    q: ref q_inner,
                    angle_deg: angle_deg_inner,
                    ..
                } = q_canon
                {
                    let angle_deg_sum = angle_deg + angle_deg_inner;
                    if angle_is_approx_zero(angle_deg_sum) {
                        *q_inner.clone()
                    } else {
                        Vector::VectorTilt {
                            q: q_inner.clone(),
                            angle_deg: canon_angle(angle_deg_sum),
                            dbg: dbg.clone(),
                        }
                    }
                } else if angle_is_approx_zero(*angle_deg) {
                    q_canon
                } else {
                    Vector::VectorTilt {
                        q: Box::new(q_canon),
                        angle_deg: canon_angle(*angle_deg),
                        dbg: dbg.clone(),
                    }
                }
            }
            Vector::UniformVectorSuperpos {
                ref q1,
                ref q2,
                ref dbg,
            } => {
                let q1_canon = q1.canonicalize();
                let q2_canon = q2.canonicalize();

                match (&q1_canon, &q2_canon) {
                    (
                        Vector::VectorTilt {
                            q: ref inner_q1,
                            angle_deg: inner_angle_deg1,
                            dbg: inner_dbg1,
                        },
                        Vector::VectorTilt {
                            q: ref inner_q2,
                            angle_deg: inner_angle_deg2,
                            ..
                        },
                    ) if on_phase(*inner_angle_deg1, *inner_angle_deg2) => Vector::VectorTilt {
                        q: Box::new(Vector::UniformVectorSuperpos {
                            q1: inner_q1.clone(),
                            q2: inner_q2.clone(),
                            dbg: dbg.clone(),
                        }),
                        angle_deg: canon_angle(*inner_angle_deg1),
                        dbg: inner_dbg1.clone(),
                    },
                    _ => Vector::UniformVectorSuperpos {
                        q1: Box::new(q1_canon),
                        q2: Box::new(q2_canon),
                        dbg: dbg.clone(),
                    },
                }
            }
            Vector::VectorTensor { ref qs, ref dbg } => {
                let vecs_canon: Vec<_> = qs.iter().map(Vector::canonicalize).collect();
                let mut new_qs = vec![];
                let mut angle_deg_sum = 0.0;
                let mut new_tilt_dbg = None;
                for vec in vecs_canon {
                    if let Vector::VectorTilt {
                        ref q,
                        angle_deg,
                        dbg: tilt_dbg,
                    } = vec
                    {
                        angle_deg_sum += angle_deg;
                        new_tilt_dbg = new_tilt_dbg.or(tilt_dbg);

                        if let Vector::VectorUnit { .. } = **q {
                            // Skip units
                        } else if let Vector::VectorTensor {
                            qs: ref inner_qs, ..
                        } = **q
                        {
                            // No need to look for tilts here because we can
                            // inductively assume they were moved to the
                            // outside and we just found them
                            new_qs.extend_from_slice(inner_qs);
                        } else {
                            new_qs.push(*q.clone());
                        }
                    } else if let Vector::VectorUnit { .. } = vec {
                        // Skip units
                    } else if let Vector::VectorTensor {
                        qs: ref inner_qs, ..
                    } = vec
                    {
                        // No need to look for tilts here because we can
                        // inductively assume they would've been moved to the
                        // outside and found above
                        new_qs.extend_from_slice(inner_qs);
                    } else {
                        new_qs.push(vec.clone());
                    }
                }

                let new_tensor = if new_qs.is_empty() {
                    Vector::VectorUnit { dbg: dbg.clone() }
                } else if new_qs.len() == 1 {
                    new_qs[0].clone()
                } else {
                    Vector::VectorTensor {
                        qs: new_qs,
                        dbg: dbg.clone(),
                    }
                };

                if angle_is_approx_zero(angle_deg_sum) {
                    new_tensor
                } else {
                    Vector::VectorTilt {
                        q: Box::new(new_tensor),
                        angle_deg: canon_angle(angle_deg_sum),
                        dbg: new_tilt_dbg.or_else(|| dbg.clone()),
                    }
                }
            }
        }
    }
}

// ----- Basis -----

#[derive(Debug, Clone, PartialEq)]
pub enum Basis {
    BasisLiteral {
        vecs: Vec<Vector>,
        dbg: Option<DebugLoc>,
    },
    EmptyBasisLiteral {
        dbg: Option<DebugLoc>,
    },
    BasisTensor {
        bases: Vec<Basis>,
        dbg: Option<DebugLoc>,
    },
}

impl Basis {
    /// Returns the source code location for this node.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            Basis::BasisLiteral { vecs: _, dbg } => dbg.clone(),
            Basis::EmptyBasisLiteral { dbg } => dbg.clone(),
            Basis::BasisTensor { bases: _, dbg } => dbg.clone(),
        }
    }

    /// Returns number of qubits represented by a basis (|b| in the Appendix)
    /// or None if any basis vector involved is malformed (see Vector::get_dim()).
    pub fn get_dim(&self) -> Option<usize> {
        match self {
            Basis::BasisLiteral { ref vecs, .. } => {
                if vecs.is_empty() {
                    None
                } else {
                    vecs[0].get_dim().and_then(|first_dim| {
                        if vecs[1..]
                            .iter()
                            .all(|vec| vec.get_dim().is_some_and(|vec_dim| vec_dim == first_dim))
                        {
                            Some(first_dim)
                        } else {
                            None
                        }
                    })
                }
            }

            Basis::EmptyBasisLiteral { .. } => Some(0),

            Basis::BasisTensor { ref bases, .. } => {
                if bases.len() < 2 {
                    None
                } else {
                    bases
                        .iter()
                        .try_fold(0, |acc, v| v.get_dim().map(|dim| acc + dim))
                }
            }
        }
    }

    /// Returns the zero-indexed qubit positions that are correspond to the
    /// provided atom. This Ξva[bv] in Appendix A. Returning None indicates a
    /// malformed vector (currently, when basis vectors in a basis literal do
    /// not have matching indices or when Vector::get_atom_indices() considers
    /// any vector malformed).
    pub fn get_atom_indices(&self, atom: VectorAtomKind) -> Option<Vec<usize>> {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
                if vecs.is_empty() {
                    None
                } else {
                    vecs[0]
                        .get_atom_indices(atom)
                        .and_then(|first_vec_indices| {
                            if vecs[1..].iter().all(|vec| {
                                vec.get_atom_indices(atom)
                                    .is_some_and(|vec_indices| vec_indices == first_vec_indices)
                            }) {
                                Some(first_vec_indices)
                            } else {
                                None
                            }
                        })
                }
            }

            Basis::EmptyBasisLiteral { .. } => Some(vec![]),

            Basis::BasisTensor { ref bases, .. } => {
                if bases.len() < 2 {
                    None
                } else {
                    let mut offset = 0;
                    let mut indices = vec![];
                    for basis in bases {
                        let basis_indices = basis.get_atom_indices(atom)?;
                        for idx in basis_indices {
                            indices.push(idx + offset);
                        }
                        let basis_dim = basis.get_dim()?;
                        offset += basis_dim;
                    }
                    Some(indices)
                }
            }
        }
    }

    /// Returns a version of this basis without any '?' or '_' atoms. Assumes
    /// the original basis is well-typed.
    pub fn make_explicit(&self) -> Basis {
        match self {
            Basis::BasisLiteral { ref vecs, ref dbg } => {
                // This is a little bit overpowered: according to the
                // orthogonality rules, a basis literal can contain at most one
                // lonely '?' or '_'. (That is, {'?'} would typecheck but {'?',
                // '?'+'?' would not.) This code will do the job, though.
                let vecs_explicit: Vec<_> = vecs
                    .iter()
                    .map(Vector::make_explicit)
                    .filter(|vec| !matches!(vec, Vector::VectorUnit { .. }))
                    .collect();
                if vecs_explicit.is_empty() {
                    Basis::EmptyBasisLiteral { dbg: dbg.clone() }
                } else {
                    Basis::BasisLiteral {
                        vecs: vecs_explicit,
                        dbg: dbg.clone(),
                    }
                }
            }

            Basis::EmptyBasisLiteral { .. } => self.clone(),

            Basis::BasisTensor { ref bases, ref dbg } => {
                let bases_explicit: Vec<_> = bases
                    .iter()
                    .map(Basis::make_explicit)
                    .filter(|basis| !matches!(basis, Basis::EmptyBasisLiteral { .. }))
                    .collect();
                // Make an assumption that this is well-formed
                if bases_explicit.is_empty() {
                    Basis::EmptyBasisLiteral { dbg: dbg.clone() }
                } else if bases_explicit.len() == 1 {
                    bases_explicit[0].clone()
                } else {
                    Basis::BasisTensor {
                        bases: bases_explicit,
                        dbg: dbg.clone(),
                    }
                }
            }
        }
    }
}

// ----- Expressions -----

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Variable {
        name: String,
        dbg: Option<DebugLoc>,
    },
    UnitLiteral {
        dbg: Option<DebugLoc>,
    },
    Adjoint {
        func: Box<Expr>,
        dbg: Option<DebugLoc>,
    },
    Pipe {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        dbg: Option<DebugLoc>,
    },
    Measure {
        basis: Basis,
        dbg: Option<DebugLoc>,
    },
    Discard {
        dbg: Option<DebugLoc>,
    },
    Tensor {
        vals: Vec<Expr>,
        dbg: Option<DebugLoc>,
    },
    BasisTranslation {
        bin: Basis,
        bout: Basis,
        dbg: Option<DebugLoc>,
    },
    Predicated {
        then_func: Box<Expr>,
        else_func: Box<Expr>,
        pred: Basis,
        dbg: Option<DebugLoc>,
    },
    NonUniformSuperpos {
        pairs: Vec<(f64, QLit)>,
        dbg: Option<DebugLoc>,
    },
    Conditional {
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        cond: Box<Expr>,
        dbg: Option<DebugLoc>,
    },
    QLit(QLit),
}

// ----- Statements -----

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign {
        lhs: String,
        rhs: Expr,
        dbg: Option<DebugLoc>,
    },
    UnpackAssign {
        lhs: Vec<String>,
        rhs: Expr,
        dbg: Option<DebugLoc>,
    },
    Return {
        val: Expr,
        dbg: Option<DebugLoc>,
    },
}

// ----- Functions -----

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<(Type, String)>,
    pub ret_type: Type,
    pub body: Vec<Stmt>,
    pub dbg: Option<DebugLoc>,
}

// ----- Program -----

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<FunctionDef>,
    pub dbg: Option<DebugLoc>,
}

// ----- Miscellaneous angle math -----

/// Tolerance for floating point comparison
const ATOL: f64 = 1e-12;

/// Returns a canon form of this angle in the range [0.0, 360.0).
pub fn canon_angle(angle_deg: f64) -> f64 {
    // angle_deg % 360 could be negative. This will always be nonnegative.
    angle_deg.rem_euclid(360.0)
}

/// Returns true if an angle is approximately 0 degrees.
pub fn angle_is_approx_zero(angle_deg: f64) -> bool {
    canon_angle(angle_deg).abs() < ATOL
}

/// Returns true iff the two phases are the same angle (up to a multiple of 360)
pub fn on_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mod360 = canon_angle(diff);
    mod360.abs() < ATOL
}

/// Returns true iff the two phases differ by 180 degrees (up to a multiple of
/// 360)
pub fn off_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mod360 = canon_angle(diff);
    (mod360 - 180.0).abs() < ATOL
}

//
// ─── UNIT TESTS ─────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod test_ast_basis;
#[cfg(test)]
mod test_ast_vec_qlit;
