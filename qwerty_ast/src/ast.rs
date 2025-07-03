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
                format!("({} @ {})", q.to_programmer_str(), angle_deg)
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
    /// (currently, if both sides of a superposition have different dimensions).
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
                if inner_bvs.len() == 0 {
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
    /// superposition have different dimensions).
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
                if inner_bvs.is_empty() {
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
                if qs.is_empty() {
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
                if bases.is_empty() {
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
    /// not have matching indices, or if any basis vector contains an empty
    /// tensor product or a superposition with indices that do not match).
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
                if bases.is_empty() {
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
