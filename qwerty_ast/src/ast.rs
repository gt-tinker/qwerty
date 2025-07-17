/*
 * QWERTY Programming Language Compiler
 * Abstract Syntax Tree (AST) Definitions
 *
 * This module defines the Abstract Syntax Tree (AST) structures
 * used for parsing and representing QWERTY programs.
 *
 */

use crate::dbg::DebugLoc;
use std::cmp::Ordering;
use std::fmt;

// ----- Types -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    FuncType { in_ty: Box<Type>, out_ty: Box<Type> },
    RevFuncType { in_out_ty: Box<Type> },
    RegType { elem_ty: RegKind, dim: u64 }, // TODO: dim: DimExpr instead of u64
    UnitType,
}

impl fmt::Display for Type {
    /// Returns a string representation of a type that matches the syntax for
    /// the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::FuncType { in_ty, out_ty } => match (&**in_ty, &**out_ty) {
                (
                    Type::RegType {
                        elem_ty: in_elem_ty,
                        dim: in_dim,
                    },
                    Type::RegType {
                        elem_ty: out_elem_ty,
                        dim: out_dim,
                    },
                ) if *in_elem_ty != RegKind::Basis && *out_elem_ty != RegKind::Basis => {
                    let prefix = match (in_elem_ty, out_elem_ty) {
                        (RegKind::Qubit, RegKind::Qubit) => "q",
                        (RegKind::Qubit, RegKind::Bit) => "qb",
                        (RegKind::Bit, RegKind::Qubit) => "bq",
                        (RegKind::Bit, RegKind::Bit) => "b",
                        (RegKind::Basis, _) | (_, RegKind::Basis) => {
                            unreachable!("bases cannot be function arguments/results")
                        }
                    };
                    write!(f, "{}func[", prefix)?;
                    if in_elem_ty == out_elem_ty && in_dim == out_dim {
                        write!(f, "{}]", in_dim)
                    } else {
                        write!(f, "{},{}]", in_dim, out_dim)
                    }
                }
                _ => write!(f, "func[{},{}]", in_ty, out_ty),
            },
            Type::RevFuncType { in_out_ty } => match &**in_out_ty {
                Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } => write!(f, "rev_qfunc[{}]", dim),
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } => write!(f, "rev_bfunc[{}]", dim),
                _ => write!(f, "rev_func[{}]", in_out_ty),
            },
            Type::RegType {
                elem_ty: RegKind::Qubit,
                dim,
            } => write!(f, "qubit[{}]", dim),
            Type::RegType {
                elem_ty: RegKind::Bit,
                dim,
            } => write!(f, "bit[{}]", dim),
            Type::RegType {
                elem_ty: RegKind::Basis,
                dim,
            } => write!(f, "basis[{}]", dim),
            Type::UnitType => write!(f, "None"),
        }
    }
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
    QubitUnit {
        dbg: Option<DebugLoc>,
    },
}

impl QLit {
    /// Converts a qubit literal to a basis vector since in the spec, every ql
    /// is a bv.
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
            QLit::QubitUnit { dbg } => Vector::VectorUnit { dbg: dbg.clone() },
        }
    }

    /// Returns an equivalent qubit literal in canon form (see
    /// Vector::canonicalize()).
    pub fn canonicalize(&self) -> QLit {
        self.convert_to_basis_vector()
            .canonicalize()
            .convert_to_qubit_literal()
            .expect(concat!(
                "converting a qlit to a basis vector should allow converting ",
                "back to a qlit"
            ))
    }
}

impl fmt::Display for QLit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLit::ZeroQubit { .. } => write!(f, "'0'"),
            QLit::OneQubit { .. } => write!(f, "'1'"),
            QLit::QubitTilt { q, angle_deg, .. } => {
                if angles_are_approx_equal(*angle_deg, 180.0) {
                    write!(f, "-{}", **q)
                } else {
                    write!(f, "({})@{}", **q, *angle_deg)
                }
            }
            QLit::UniformSuperpos { q1, q2, .. } => write!(f, "({}) + ({})", **q1, **q2),
            QLit::QubitTensor { qs, .. } => {
                for (i, q) in qs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", q)?;
                }
                Ok(())
            }
            QLit::QubitUnit { .. } => write!(f, "''"),
        }
    }
}

// ----- Vector -----

/// Represents either a padding ('?') or a target ('_') vector atom. This is
/// "va" in the spec, except limited to the variants we actually use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAtomKind {
    PadAtom,    // '?'
    TargetAtom, // '_'
}

impl fmt::Display for VectorAtomKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorAtomKind::PadAtom => write!(f, "'?'"),
            VectorAtomKind::TargetAtom => write!(f, "'_'"),
        }
    }
}

//#[derive(Debug, Clone, PartialOrd, PartialEq, Eq)]
#[derive(Debug, Clone)]
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
    /// Converts this vector into a qubit literal. Returns None if this vector
    /// contains any padding or target atoms.
    pub fn convert_to_qubit_literal(&self) -> Option<QLit> {
        match self {
            Vector::ZeroVector { dbg } => Some(QLit::ZeroQubit { dbg: dbg.clone() }),
            Vector::OneVector { dbg } => Some(QLit::OneQubit { dbg: dbg.clone() }),
            Vector::VectorTilt { q, angle_deg, dbg } => {
                (**q).convert_to_qubit_literal().map(|qlq| QLit::QubitTilt {
                    q: Box::new(qlq),
                    angle_deg: *angle_deg,
                    dbg: dbg.clone(),
                })
            }
            Vector::UniformVectorSuperpos { q1, q2, dbg } => (**q1)
                .convert_to_qubit_literal()
                .zip((**q2).convert_to_qubit_literal())
                .map(|(qlq1, qlq2)| QLit::UniformSuperpos {
                    q1: Box::new(qlq1),
                    q2: Box::new(qlq2),
                    dbg: dbg.clone(),
                }),
            Vector::VectorTensor { qs, dbg } => qs
                .iter()
                .map(Vector::convert_to_qubit_literal)
                .collect::<Option<Vec<QLit>>>()
                .map(|qlqs| QLit::QubitTensor {
                    qs: qlqs,
                    dbg: dbg.clone(),
                }),
            Vector::VectorUnit { dbg } => Some(QLit::QubitUnit { dbg: dbg.clone() }),

            Vector::PadVector { .. } | Vector::TargetVector { .. } => None,
        }
    }

    /// Returns a version of this vector with no debug info. Useful for
    /// comparing vectors/bases without considering debug info.
    pub fn strip_dbg(&self) -> Vector {
        match self {
            Vector::ZeroVector { .. } => Vector::ZeroVector { dbg: None },
            Vector::OneVector { .. } => Vector::OneVector { dbg: None },
            Vector::PadVector { .. } => Vector::PadVector { dbg: None },
            Vector::TargetVector { .. } => Vector::TargetVector { dbg: None },
            Vector::VectorTilt { q, angle_deg, .. } => Vector::VectorTilt {
                q: Box::new(q.strip_dbg()),
                angle_deg: *angle_deg,
                dbg: None,
            },
            Vector::UniformVectorSuperpos { q1, q2, .. } => Vector::UniformVectorSuperpos {
                q1: Box::new(q1.strip_dbg()),
                q2: Box::new(q2.strip_dbg()),
                dbg: None,
            },
            Vector::VectorTensor { qs, .. } => Vector::VectorTensor {
                qs: qs.iter().map(Vector::strip_dbg).collect(),
                dbg: None,
            },
            Vector::VectorUnit { .. } => Vector::VectorUnit { dbg: None },
        }
    }

    /// Returns number of non-target and non-padding qubits represented by a basis
    /// vector (⌊bv⌋ in the spec) or None if the basis vector is malformed
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
    /// padding and target qubits. (This is |bv| in the spec.) Returns None
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
    /// provided atom. This Ξva[bv] in the spec. Returning None indicates a
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

            (Vector::VectorTensor { qs, .. }, _) => {
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
            Vector::PadVector { dbg } | Vector::TargetVector { dbg } => {
                Vector::VectorUnit { dbg: dbg.clone() }
            }
            Vector::VectorTilt { q, angle_deg, dbg } => {
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
            Vector::UniformVectorSuperpos { q1, q2, dbg } => Vector::UniformVectorSuperpos {
                q1: Box::new(q1.make_explicit()),
                q2: Box::new(q2.make_explicit()),
                dbg: dbg.clone(),
            },
            Vector::VectorTensor { qs, dbg } => {
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

    /// Suppose left=(l1, l2) and right=(r1, r2). Then this function considers
    /// (l1 + l2) + (r1 + r2). If r1 = l1@180 and r2 = l2, up to some
    /// commutation of the terms of the inner superpositions, this returns two
    /// vectors (cons, dest). cons is the constructively interfering vector
    /// (r2/l2), and dest was the destructively interfering vector (l1).
    fn try_interference<'v>(
        left: (&'v Vector, &'v Vector),
        right: (&'v Vector, &'v Vector),
    ) -> Option<(&'v Vector, &'v Vector)> {
        match (left, right) {
            (
                (q1l, q2l),
                (
                    q1r,
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                ),
            )
            | (
                (q1l, q2l),
                (
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                    q1r,
                ),
            )
            | (
                (q2l, q1l),
                (
                    q1r,
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                ),
            )
            | (
                (q2l, q1l),
                (
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                    q1r,
                ),
            ) if q1l == q1r && q2l == &**bq2r && angles_are_approx_equal(*angle_deg, 180.0) => {
                Some((q1l, q2l))
            }

            _ => None,
        }
    }

    /// Simplifies the vector `left+right` by performing interference.
    fn interfere(left: &Vector, right: &Vector) -> Option<Vector> {
        match (left, right) {
            (
                Vector::UniformVectorSuperpos {
                    q1: lq1, q2: lq2, ..
                },
                Vector::UniformVectorSuperpos {
                    q1: rq1, q2: rq2, ..
                },
            ) => Vector::try_interference((&**lq1, &**lq2), (&**rq1, &**rq2))
                .or_else(|| Vector::try_interference((&**rq1, &**rq2), (&**lq1, &**lq2)))
                .map(|(cons, _dest)| cons.clone()),

            (
                Vector::UniformVectorSuperpos {
                    q1: lq1, q2: lq2, ..
                },
                Vector::VectorTilt {
                    q: rq,
                    angle_deg: rangle_deg,
                    dbg,
                },
            )
            | (
                Vector::VectorTilt {
                    q: rq,
                    angle_deg: rangle_deg,
                    dbg,
                },
                Vector::UniformVectorSuperpos {
                    q1: lq1, q2: lq2, ..
                },
            ) if angles_are_approx_equal(*rangle_deg, 180.0) => match &**rq {
                Vector::UniformVectorSuperpos {
                    q1: rq1, q2: rq2, ..
                } => Vector::try_interference((&**lq1, &**lq2), (&**rq1, &**rq2))
                    .map(|(_cons, dest)| dest.clone())
                    .or_else(|| {
                        Vector::try_interference((&**rq1, &**rq2), (&**lq1, &**lq2)).map(
                            |(_cons, dest)| Vector::VectorTilt {
                                q: Box::new(dest.clone()),
                                angle_deg: 180.0,
                                dbg: dbg.clone(),
                            },
                        )
                    }),

                _ => None,
            },

            _ => None,
        }
    }

    /// Returns an equivalent vector such that:
    /// 1. No tensor product contains a []
    /// 2. No tensor product contains another tensor product
    /// 3. All @s are propagated to the outside
    /// 4. All tensor product angles in canon form (i.e., in the interval
    ///    [0.0, 360.0))
    /// 5. The terms q1 and q2 of a superposition q1+q2 are ordered such that
    ///    q1 <= q2.
    /// 6. Basic interference is performed:
    ///        (q1 + q2) + (q1 - q2) -> q1
    ///    and
    ///        (q1 + q2) + -(q1 - q2) -> q1
    /// Assumes this vector is well-typed.
    pub fn canonicalize(&self) -> Vector {
        match self {
            Vector::ZeroVector { .. }
            | Vector::OneVector { .. }
            | Vector::PadVector { .. }
            | Vector::TargetVector { .. }
            | Vector::VectorUnit { .. } => self.clone(),

            Vector::VectorTilt { q, angle_deg, dbg } => {
                let q_canon = q.canonicalize();
                if let Vector::VectorTilt {
                    q: q_inner,
                    angle_deg: angle_deg_inner,
                    ..
                } = q_canon
                {
                    let angle_deg_sum = angle_deg + angle_deg_inner;
                    if angle_is_approx_zero(canon_angle(angle_deg_sum)) {
                        *q_inner.clone()
                    } else {
                        Vector::VectorTilt {
                            q: q_inner.clone(),
                            angle_deg: canon_angle(angle_deg_sum),
                            dbg: dbg.clone(),
                        }
                    }
                } else if angle_is_approx_zero(canon_angle(*angle_deg)) {
                    q_canon
                } else {
                    Vector::VectorTilt {
                        q: Box::new(q_canon),
                        angle_deg: canon_angle(*angle_deg),
                        dbg: dbg.clone(),
                    }
                }
            }

            Vector::UniformVectorSuperpos { q1, q2, dbg } => {
                let q1_canon = q1.canonicalize();
                let q2_canon = q2.canonicalize();

                let (first_q, second_q) = if q2_canon < q1_canon {
                    (q2_canon, q1_canon)
                } else {
                    (q1_canon, q2_canon)
                };

                if let Some(vec) = Vector::interfere(&first_q, &second_q) {
                    // Call canonicalize() here in case there is
                    // e.g. a nested tilt
                    vec.canonicalize()
                } else {
                    Vector::UniformVectorSuperpos {
                        q1: Box::new(first_q),
                        q2: Box::new(second_q),
                        dbg: dbg.clone(),
                    }
                }
            }

            Vector::VectorTensor { qs, dbg } => {
                let vecs_canon: Vec<_> = qs.iter().map(Vector::canonicalize).collect();
                let mut new_qs = vec![];
                let mut angle_deg_sum = 0.0;
                let mut new_tilt_dbg = None;
                for vec in &vecs_canon {
                    if let Vector::VectorTilt {
                        q,
                        angle_deg,
                        dbg: tilt_dbg,
                    } = vec
                    {
                        angle_deg_sum += angle_deg;
                        new_tilt_dbg = new_tilt_dbg.or_else(|| tilt_dbg.clone());

                        if let Vector::VectorUnit { .. } = **q {
                            // Skip units
                        } else if let Vector::VectorTensor { qs: inner_qs, .. } = &**q {
                            // No need to look for tilts here because we can
                            // inductively assume they were moved to the
                            // outside and we just found them
                            new_qs.extend_from_slice(inner_qs);
                        } else {
                            new_qs.push(*q.clone());
                        }
                    } else if let Vector::VectorUnit { .. } = vec {
                        // Skip units
                    } else if let Vector::VectorTensor { qs: inner_qs, .. } = vec {
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

                if angle_is_approx_zero(canon_angle(angle_deg_sum)) {
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

    /// Removes phases (except those directly inside superpositions). This is
    /// intended for use in span equivalence checking. The vector should be
    /// type-checked and canonicalized first.
    pub fn normalize(&self) -> Vector {
        match self {
            Vector::VectorTilt { q, .. } => (**q).clone(),
            _ => self.clone(),
        }
    }
}

impl Ord for Vector {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Vector::VectorUnit { dbg: dbgl }, Vector::VectorUnit { dbg: dbgr })
            | (Vector::ZeroVector { dbg: dbgl }, Vector::ZeroVector { dbg: dbgr })
            | (Vector::OneVector { dbg: dbgl }, Vector::OneVector { dbg: dbgr })
            | (Vector::PadVector { dbg: dbgl }, Vector::PadVector { dbg: dbgr })
            | (Vector::TargetVector { dbg: dbgl }, Vector::TargetVector { dbg: dbgr }) => {
                dbgl.cmp(dbgr)
            }

            // [] is the least element
            (Vector::VectorUnit { .. }, _) => Ordering::Less,
            (_, Vector::VectorUnit { .. }) => Ordering::Greater,

            // '0' is the next least
            (Vector::ZeroVector { .. }, _) => Ordering::Less,
            (_, Vector::ZeroVector { .. }) => Ordering::Greater,

            // Then '1' is the next least
            (Vector::OneVector { .. }, _) => Ordering::Less,
            (_, Vector::OneVector { .. }) => Ordering::Greater,

            // Then '?' is the next least
            (Vector::PadVector { .. }, _) => Ordering::Less,
            (_, Vector::PadVector { .. }) => Ordering::Greater,

            // Then '_' is the next least
            (Vector::TargetVector { .. }, _) => Ordering::Less,
            (_, Vector::TargetVector { .. }) => Ordering::Greater,

            // Then @
            (
                Vector::VectorTilt {
                    q: ql,
                    angle_deg: angle_degl,
                    dbg: dbgl,
                },
                Vector::VectorTilt {
                    q: qr,
                    angle_deg: angle_degr,
                    dbg: dbgr,
                },
            ) => ql
                .cmp(qr)
                .then(angle_approx_total_cmp(*angle_degl, *angle_degr))
                .then(dbgl.cmp(dbgr)),
            (Vector::VectorTilt { .. }, _) => Ordering::Less,
            (_, Vector::VectorTilt { .. }) => Ordering::Greater,

            // Then +
            (
                Vector::UniformVectorSuperpos {
                    q1: q1l,
                    q2: q2l,
                    dbg: dbgl,
                },
                Vector::UniformVectorSuperpos {
                    q1: q1r,
                    q2: q2r,
                    dbg: dbgr,
                },
            ) => q1l.cmp(q1r).then(q2l.cmp(q2r)).then(dbgl.cmp(dbgr)),
            (Vector::UniformVectorSuperpos { .. }, _) => Ordering::Less,
            (_, Vector::UniformVectorSuperpos { .. }) => Ordering::Greater,

            // Then *
            (
                Vector::VectorTensor { qs: qsl, dbg: dbgl },
                Vector::VectorTensor { qs: qsr, dbg: dbgr },
            ) => qsl
                .iter()
                .zip(qsr.iter())
                .fold(Ordering::Equal, |acc, (l, r)| acc.then(l.cmp(r)))
                .then(dbgl.cmp(dbgr)),
        }
    }
}

impl PartialOrd for Vector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Unfortunately, we can't rely on #[derive] to implement this because NaN !=
// NaN unless we intentionally use f64::total_cmp() as we do below.
impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Vector::ZeroVector { dbg: dbgl }, Vector::ZeroVector { dbg: dbgr }) => dbgl == dbgr,
            (Vector::OneVector { dbg: dbgl }, Vector::OneVector { dbg: dbgr }) => dbgl == dbgr,
            (Vector::PadVector { dbg: dbgl }, Vector::PadVector { dbg: dbgr }) => dbgl == dbgr,
            (Vector::TargetVector { dbg: dbgl }, Vector::TargetVector { dbg: dbgr }) => {
                dbgl == dbgr
            }
            (
                Vector::VectorTilt {
                    q: ql,
                    angle_deg: angle_degl,
                    dbg: dbgl,
                },
                Vector::VectorTilt {
                    q: qr,
                    angle_deg: angle_degr,
                    dbg: dbgr,
                },
            ) => ql == qr && angles_are_approx_equal(*angle_degl, *angle_degr) && dbgl == dbgr,
            (
                Vector::UniformVectorSuperpos {
                    q1: q1l,
                    q2: q2l,
                    dbg: dbgl,
                },
                Vector::UniformVectorSuperpos {
                    q1: q1r,
                    q2: q2r,
                    dbg: dbgr,
                },
            ) => q1l == q1r && q2l == q2r && dbgl == dbgr,
            (
                Vector::VectorTensor { qs: qsl, dbg: dbgl },
                Vector::VectorTensor { qs: qsr, dbg: dbgr },
            ) => qsl == qsr && dbgl == dbgr,
            (Vector::VectorUnit { dbg: dbgl }, Vector::VectorUnit { dbg: dbgr }) => dbgl == dbgr,
            _ => false,
        }
    }
}

impl Eq for Vector {}

impl fmt::Display for Vector {
    /// Represents a vector in a human-readable form for error messages sent
    /// back to the programmer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vector::ZeroVector { .. } => write!(f, "'0'"),
            Vector::OneVector { .. } => write!(f, "'1'"),
            Vector::PadVector { .. } => write!(f, "'?'"),
            Vector::TargetVector { .. } => write!(f, "'_'"),
            Vector::VectorTilt { q, angle_deg, .. } => {
                if angles_are_approx_equal(*angle_deg, 180.0) {
                    write!(f, "-{}", **q)
                } else {
                    write!(f, "({})@{}", **q, *angle_deg)
                }
            }
            Vector::UniformVectorSuperpos { q1, q2, .. } => write!(f, "({}) + ({})", **q1, **q2),
            Vector::VectorTensor { qs, .. } => {
                for (i, q) in qs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", q)?;
                }
                Ok(())
            }
            Vector::VectorUnit { .. } => write!(f, "''"),
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
    /// Returns the n-qubit standard basis, where n = dim.
    pub fn std(dim: usize, dbg: Option<DebugLoc>) -> Basis {
        let mut bases = vec![];
        for _ in 0..dim {
            bases.push(Basis::BasisLiteral {
                vecs: vec![
                    Vector::ZeroVector { dbg: dbg.clone() },
                    Vector::OneVector { dbg: dbg.clone() },
                ],
                dbg: dbg.clone(),
            });
        }
        Basis::BasisTensor { bases, dbg: dbg }
    }

    /// Returns a version of this basis with no debug info. Useful for
    /// comparing vectors/bases without considering debug info.
    pub fn strip_dbg(&self) -> Basis {
        match self {
            Basis::BasisLiteral { vecs, .. } => Basis::BasisLiteral {
                vecs: vecs.iter().map(Vector::strip_dbg).collect(),
                dbg: None,
            },
            Basis::EmptyBasisLiteral { .. } => Basis::EmptyBasisLiteral { dbg: None },
            Basis::BasisTensor { bases, .. } => Basis::BasisTensor {
                bases: bases.iter().map(Basis::strip_dbg).collect(),
                dbg: None,
            },
        }
    }

    /// Returns the source code location for this node.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            Basis::BasisLiteral { vecs: _, dbg } => dbg.clone(),
            Basis::EmptyBasisLiteral { dbg } => dbg.clone(),
            Basis::BasisTensor { bases: _, dbg } => dbg.clone(),
        }
    }

    /// Returns number of qubits represented by a basis (|b| in the spec)
    /// or None if any basis vector involved is malformed (see Vector::get_dim()).
    pub fn get_dim(&self) -> Option<usize> {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
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

            Basis::BasisTensor { bases, .. } => {
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
    /// provided atom. This Ξva[bv] in the spec. Returning None indicates a
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

            Basis::BasisTensor { bases, .. } => {
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
            Basis::BasisLiteral { vecs, dbg } => {
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

            Basis::BasisTensor { bases, dbg } => {
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

    /// Returns an equivalent basis such that:
    /// 1. All vectors are in the form promised by Vector::canonicalize()
    /// 2. No tensor product contains a []
    /// 3. No tensor product contains another tensor product
    /// 4. No basis literal consistingly only of an empty vector, {[]}, are
    ///    replaced with an empty basis literal []
    pub fn canonicalize(&self) -> Basis {
        match self {
            Basis::EmptyBasisLiteral { .. } => self.clone(),

            Basis::BasisLiteral { vecs, dbg } => {
                let canon_vecs: Vec<_> = vecs.iter().map(Vector::canonicalize).collect();
                if canon_vecs.len() == 1 && matches!(&canon_vecs[0], Vector::VectorUnit { .. }) {
                    Basis::EmptyBasisLiteral { dbg: dbg.clone() }
                } else {
                    Basis::BasisLiteral {
                        vecs: canon_vecs,
                        dbg: dbg.clone(),
                    }
                }
            }

            Basis::BasisTensor { bases, dbg } => {
                let bases_canon: Vec<_> = bases
                    .iter()
                    .map(Basis::canonicalize)
                    .flat_map(|basis| {
                        match basis {
                            Basis::BasisLiteral { .. } => vec![basis],
                            Basis::EmptyBasisLiteral { .. } => vec![],
                            Basis::BasisTensor { bases: inner_bases, .. } => inner_bases,
                        }
                    })
                    .collect();
                if bases_canon.is_empty() {
                    Basis::EmptyBasisLiteral { dbg: dbg.clone() }
                } else if bases_canon.len() == 1 {
                    bases_canon[0].clone()
                } else {
                    Basis::BasisTensor {
                        bases: bases_canon,
                        dbg: dbg.clone(),
                    }
                }
            }
        }
    }

    /// Converts to a vector of basis elements in order.
    fn to_vec(&self) -> Vec<Basis> {
        match self {
            Basis::EmptyBasisLiteral { .. } => vec![],

            Basis::BasisLiteral { .. } => vec![self.clone()],

            Basis::BasisTensor { bases, .. } => bases.clone(),
        }
    }

    /// Converts to a stack of basis elements such that the first element
    /// popped is the leftmost.
    pub fn to_stack(&self) -> Vec<Basis> {
        let mut vec = self.to_vec();
        vec.reverse();
        vec
    }

    /// Returns true if this basis fully spans the n-qubit space (if
    /// self.get_dim() == n). Behavior is undefined if this basis is not
    /// well-typed.
    pub fn fully_spans(&self) -> bool {
        match self {
            // A matter of definition, but let's say no. You can't write
            // [] >> [] anyway.
            Basis::EmptyBasisLiteral { .. } => false,

            Basis::BasisLiteral { vecs, .. } => {
                if let Some(dim) = self.get_dim() {
                    // Assumes that all vectors are mutually orthogonal, as
                    // checked by type checking.
                    // TODO: avoid panicking when there are more than 4.3
                    //       billion vectors in a basis literal
                    equals_2_to_the_n(dim, vecs.len().try_into().unwrap())
                } else {
                    // Conservatively assume not
                    false
                }
            }

            Basis::BasisTensor { bases, .. } => bases.iter().all(Basis::fully_spans),
        }
    }

    /// Removes phases (except those directly inside superpositions) and sorts
    /// all vectors in basis literals. The resulting basis has the same span
    /// but is not necessarily equivalent. This is intended for use in span
    /// equivalence checking. The basis should be type-checked and
    /// canonicalized first.
    pub fn normalize(&self) -> Basis {
        match self {
            Basis::EmptyBasisLiteral { .. } => self.clone(),

            Basis::BasisLiteral { vecs, dbg } => {
                let mut norm_vecs: Vec<_> = vecs.iter().map(Vector::normalize).collect();
                norm_vecs.sort();
                Basis::BasisLiteral {
                    vecs: norm_vecs,
                    dbg: dbg.clone(),
                }
            }

            Basis::BasisTensor { bases, dbg } => Basis::BasisTensor {
                bases: bases.iter().map(Basis::normalize).collect(),
                dbg: dbg.clone(),
            },
        }
    }
}

impl fmt::Display for Basis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
                write!(f, "{{")?;
                for (i, vec) in vecs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", vec)?;
                }
                write!(f, "}}")
            }
            Basis::EmptyBasisLiteral { .. } => write!(f, "{{}}"),
            Basis::BasisTensor { bases, .. } => {
                for (i, b) in bases.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", b)?;
                }
                Ok(())
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
    QLit {
        qlit: QLit,
        dbg: Option<DebugLoc>,
    },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Variable { name, .. } => write!(f, "{}", name),
            Expr::UnitLiteral { .. } => write!(f, "[]"),
            Expr::Adjoint { func, .. } => write!(f, "~({})", **func),
            Expr::Pipe { lhs, rhs, .. } => write!(f, "({}) | ({})", **lhs, **rhs),
            Expr::Measure { basis, .. } => write!(f, "({}).measure", basis),
            Expr::Discard { .. } => write!(f, "discard"),
            Expr::Tensor { vals, .. } => {
                for (i, val) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "{}", val)?;
                }
                Ok(())
            }
            Expr::BasisTranslation { bin, bout, .. } => write!(f, "({}) >> ({})", bin, bout),
            Expr::Predicated {
                then_func,
                else_func,
                pred,
                ..
            } => write!(f, "({}) if ({}) else ({})", then_func, pred, else_func),
            Expr::NonUniformSuperpos { pairs, .. } => {
                for (i, (prob, qlit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}*({})", prob, qlit)?;
                }
                Ok(())
            }
            Expr::Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            } => write!(f, "({}) if ({}) else ({})", then_expr, cond, else_expr),
            Expr::QLit { qlit, .. } => write!(f, "{}", qlit),
        }
    }
}

// ----- Statements -----

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr {
        expr: Expr,
        dbg: Option<DebugLoc>,
    },
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
    pub is_rev: bool,
    pub dbg: Option<DebugLoc>,
}

impl FunctionDef {
    pub fn new(
        name: String,
        args: Vec<(Type, String)>,
        ret_type: Type,
        body: Vec<Stmt>,
        is_rev: bool, // passed from the parser
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dbg,
        }
    }

    /// Returns true if the function was explicitly annotated as reversible.
    pub fn is_reversible(&self) -> bool {
        self.is_rev
    }

    /// Reconstructs the full function type (FuncType or RevFuncType) from the
    /// FunctionDef's arguments, value return type, and reversibility flag.
    pub fn get_type(&self) -> Type {
        let in_ty = if self.args.is_empty() {
            Type::UnitType
        } else if self.args.len() == 1 {
            self.args[0].0.clone()
        } else {
            // TODO: For now, if multiple arguments are present and TupleType is not used,
            // we take the type of the first argument. This needs to be refined
            // when proper multi-argument function types are introduced (e.g. via TupleType).
            // TODO: Should fail? Ask Austin
            self.args[0].0.clone()
        };

        if self.is_rev {
            Type::RevFuncType {
                in_out_ty: Box::new(self.ret_type.clone()),
            }
        } else {
            Type::FuncType {
                in_ty: Box::new(in_ty),
                out_ty: Box::new(self.ret_type.clone()),
            }
        }
    }
}

// ----- Program -----

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<FunctionDef>,
    pub dbg: Option<DebugLoc>,
}

// ----- Miscellaneous math for angles and bits -----

/// Tolerance for floating point comparison
const ATOL: f64 = 1e-12;

/// Returns a canon form of this angle in the range [0.0, 360.0).
pub fn canon_angle(angle_deg: f64) -> f64 {
    // angle_deg % 360 could be negative. This will always be nonnegative.
    angle_deg.rem_euclid(360.0)
}

/// Returns true if two angles are approximately equal.
pub fn angles_are_approx_equal(angle_deg1: f64, angle_deg2: f64) -> bool {
    (angle_deg1 - angle_deg2).abs() < ATOL
}

/// Returns true if an angle is approximately 0 degrees.
pub fn angle_is_approx_zero(angle_deg: f64) -> bool {
    angles_are_approx_equal(angle_deg, 0.0)
}

/// Returns true iff the two phases are the same angle (up to a multiple of 360)
pub fn in_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mod360 = canon_angle(diff);
    mod360.abs() < ATOL
}

/// Returns true iff the two phases differ by 180 degrees (up to a multiple of
/// 360)
pub fn anti_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mod360 = canon_angle(diff);
    (mod360 - 180.0).abs() < ATOL
}

/// A total ordering of angles that accounts for floats being noisy
pub fn angle_approx_total_cmp(angle_deg1: f64, angle_deg2: f64) -> Ordering {
    if angles_are_approx_equal(angle_deg1, angle_deg2) {
        Ordering::Equal
    } else if angle_deg1 < angle_deg2 {
        Ordering::Less
    } else {
        // angle_deg1 > angle_deg2
        Ordering::Greater
    }
}

/// Returns true iff num == 2**n.
pub fn equals_2_to_the_n(num: usize, n: u32) -> bool {
    num.count_ones() == 1 && num.trailing_zeros() == n
}

//
// ─── UNIT TESTS ─────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod test_ast_basis;
#[cfg(test)]
mod test_ast_vec_qlit;
