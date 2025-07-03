//! Qwerty typechecker implementation: walks the AST and enforces all typing rules.

use crate::ast::*;
use crate::error::{TypeError, TypeErrorKind};
use std::collections::HashMap;
use std::iter::zip;

//
// ─── TYPE ENVIRONMENT ───────────────────────────────────────────────────────────
//

/// Tracks variable bindings (and potentially functions, quantum registers, etc.)
#[derive(Debug, Clone)]
pub struct TypeEnv {
    vars: HashMap<String, Type>,
    // TODO: Extend as needed (functions, modules, scopes, etc.)
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    // Allows Shadowing
    pub fn insert_var(&mut self, name: &str, typ: Type) {
        self.vars.insert(name.to_string(), typ);
    }

    pub fn get_var(&self, name: &str) -> Option<&Type> {
        self.vars.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
}

//
// ─── TOP-LEVEL TYPECHECKER ──────────────────────────────────────────────────────
//

/// Entry point: checks the whole program.
/// Returns Ok(()) if well-typed, or a TypeError at the first mistake (Fail fast!!)
/// TODO: (Future-work!) Change it to Multi/Batch Error reporting Result<(), Vec<TypeError>>
pub fn typecheck_program(prog: &Program) -> Result<(), TypeError> {
    for func in &prog.funcs {
        typecheck_function(func)?;
    }
    Ok(())
}

/// Typechecks a single function and its body.
pub fn typecheck_function(func: &FunctionDef) -> Result<(), TypeError> {
    let mut env = TypeEnv::new();

    // Bind function arguments in the environment.
    for (ty, name) in &func.args {
        env.insert_var(name, ty.clone());
    }

    // Track the expected return type (for return statements).
    let expected_ret_type = &func.ret_type;

    // Typecheck each statement.
    for stmt in &func.body {
        typecheck_stmt(stmt, &mut env, expected_ret_type)?;
    }

    Ok(())
}

//
// ─── STATEMENTS ────────────────────────────────────────────────────────────────
//

/// Typecheck a statement.
/// - env: The current variable/type environment.
/// - expected_ret_type: Used to check Return statements.
pub fn typecheck_stmt(
    stmt: &Stmt,
    env: &mut TypeEnv,
    expected_ret_type: &Type,
) -> Result<(), TypeError> {
    match stmt {
        Stmt::Assign { lhs, rhs, dbg: _ } => {
            let rhs_ty = typecheck_expr(rhs, env)?;
            env.insert_var(lhs, rhs_ty); // Shadowing allowed for now.
            Ok(())
        }

        // UnpackAssign checks:
        // 1. RHS must be a register type
        // 2. Number of LHS variables must match register dimension
        // 3. Each LHS variable gets typed as single-element register of same kind "RegType{ elem_ty, dim: 1 }"
        Stmt::UnpackAssign { lhs, rhs, dbg } => {
            let rhs_ty = typecheck_expr(rhs, env)?;

            match rhs_ty {
                Type::RegType { elem_ty, dim } => {
                    if lhs.len() as u64 != dim {
                        return Err(TypeError {
                            kind: TypeErrorKind::WrongArity {
                                expected: dim as usize,
                                found: lhs.len(),
                            },
                            dbg: dbg.clone(),
                        });
                    }
                    for var in lhs {
                        env.insert_var(
                            var,
                            Type::RegType {
                                elem_ty: elem_ty.clone(),
                                dim: 1,
                            },
                        );
                    }
                    Ok(())
                }
                _ => Err(TypeError {
                    kind: TypeErrorKind::InvalidType(format!(
                        "Can only unpack from register type, found: {:?}",
                        rhs_ty
                    )),
                    dbg: dbg.clone(),
                }),
            }
        }

        Stmt::Return { val, dbg } => {
            let val_ty = typecheck_expr(val, env)?;
            if &val_ty != expected_ret_type {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", expected_ret_type),
                        found: format!("{:?}", val_ty),
                    },
                    dbg: dbg.clone(),
                });
            }
            Ok(())
        }
    }
}

//
// ─── EXPRESSIONS ────────────────────────────────────────────────────────────────
//

/// Typecheck an expression and return its type.
pub fn typecheck_expr(expr: &Expr, env: &mut TypeEnv) -> Result<Type, TypeError> {
    match expr {
        Expr::Variable { name, dbg } => env.get_var(name).cloned().ok_or(TypeError {
            kind: TypeErrorKind::UndefinedVariable(name.clone()),
            dbg: dbg.clone(),
        }),

        Expr::UnitLiteral { dbg: _ } => Ok(Type::UnitType),

        Expr::Adjoint { func, dbg: _ } => {
            // Adjoint should be a function type (unitary/quantum), not classical.
            let func_ty = typecheck_expr(func, env)?;
            // TODO: Enforce Qwerty adjoint typing rules.
            Ok(func_ty)
        }

        Expr::Pipe { lhs, rhs, dbg: _ } => {
            // Typing rule: lhs type must match rhs function input type.
            let lhs_ty = typecheck_expr(lhs, env)?;
            let rhs_ty = typecheck_expr(rhs, env)?;
            match &rhs_ty {
                Type::FuncType { in_ty, out_ty } => {
                    if **in_ty != lhs_ty {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: format!("{:?}", in_ty),
                                found: format!("{:?}", lhs_ty),
                            },
                            dbg: None,
                        });
                    }
                    Ok((**out_ty).clone())
                }
                _ => Err(TypeError {
                    kind: TypeErrorKind::NotCallable(format!("{:?}", rhs_ty)),
                    dbg: None,
                }),
            }
        }

        Expr::Measure { basis, dbg: _ } => {
            // Qwerty: measurement returns classical result; basis must be valid.
            typecheck_basis(basis, env)?; //  is it a legal quantum basis?

            Ok(Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 1, // TODO: Make dynamic based on basis (Check with Austin about its validation)
                        // Self Note (verify with Austin): Currently this measurement returns a single classical bit
                        // But in real quantum programs, we might measure multiple qubits at once (e.g measuring a register of 3 qubits gives you 3 classical bits, ryt?)
                        // The number of bits returned should depend on the size/dimension of the basis being measured
                        // So, make 'dim' reflect the actual number of qubits measured, as determined by the basis argument
                        // Need to understand and work on dimensions.. Discuss bro!
            })
        }

        Expr::Discard { dbg: _ } => Ok(Type::UnitType),

        // A tensor (often a tensor product) means combining multiple quantum states or registers into a larger, composite system
        Expr::Tensor { vals, dbg: _ } => {
            // => All sub-expressions in a tensor must have the same type (e.g. all are qubits, or all are bits)
            // Tensor([Qubit, Qubit, Qubit])
            let mut t = None;
            for v in vals {
                let vty = typecheck_expr(v, env)?;

                if let Some(prev) = &t {
                    if &vty != prev {
                        // if current_type doesn't match with prev_type, error!
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: format!("{:?}", prev),
                                found: format!("{:?}", vty),
                            },
                            dbg: None,
                        });
                    }
                }
                t = Some(vty);
            }
            Ok(t.unwrap_or(Type::UnitType))
        }

        Expr::BasisTranslation { bin, bout, dbg } => {
            let left_ty = typecheck_basis(bin, env)?;
            let right_ty = typecheck_basis(bout, env)?;

            let result_ty = if let Type::RegType {
                elem_ty: RegKind::Basis,
                dim,
            } = left_ty
            {
                if dim == 0 {
                    return Err(TypeError {
                        kind: TypeErrorKind::EmptyLiteral,
                        dbg: bin.get_dbg(),
                    });
                }

                // Type of this basis translation (pending further checks)
                Type::RevFuncType {
                    in_out_ty: Box::new(Type::RegType {
                        elem_ty: RegKind::Qubit,
                        dim: dim,
                    }),
                }
            } else {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidBasis,
                    dbg: bin.get_dbg(),
                });
            };

            if left_ty != right_ty {
                return Err(TypeError {
                    kind: TypeErrorKind::DimMismatch,
                    dbg: dbg.clone(),
                });
            }

            Ok(result_ty)
        }

        Expr::Predicated {
            then_func,
            else_func,
            pred,
            dbg: _,
        } => {
            let t_ty = typecheck_expr(then_func, env)?;
            let e_ty = typecheck_expr(else_func, env)?;
            let _pred_ty = typecheck_basis(pred, env)?;
            if t_ty != e_ty {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t_ty),
                        found: format!("{:?}", e_ty),
                    },
                    dbg: None,
                });
            }
            Ok(t_ty)
        }

        Expr::NonUniformSuperpos { pairs, dbg: _ } => {
            // Each pair is (weight, QLit). All QLits must have same type.
            let mut qt = None;
            for (_, qlit) in pairs {
                let qlit_ty = typecheck_qlit(qlit, env)?;
                if let Some(prev) = &qt {
                    if &qlit_ty != prev {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: format!("{:?}", prev),
                                found: format!("{:?}", qlit_ty),
                            },
                            dbg: None,
                        });
                    }
                }
                qt = Some(qlit_ty);
            }
            Ok(qt.unwrap_or(Type::UnitType))
        }

        Expr::Conditional {
            then_expr,
            else_expr,
            cond,
            dbg: _,
        } => {
            let t_ty = typecheck_expr(then_expr, env)?;
            let e_ty = typecheck_expr(else_expr, env)?;
            let _c_ty = typecheck_expr(cond, env)?;
            if t_ty != e_ty {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t_ty),
                        found: format!("{:?}", e_ty),
                    },
                    dbg: None,
                });
            }
            Ok(t_ty)
        }

        Expr::QLit(qlit) => typecheck_qlit(qlit, env),
    }
}

//
// ─── QLIT, VECTOR, BASIS HELPERS ──────────────────────────────────────────────
//

/// Implements the O-Shuf and O-Tens rules by scanning a pair of tensor
/// products elementwise for any pair of orthogonal vectors. If it finds one,
/// that pair is sufficient to conclude that the overall tensor products are
/// orthogonal. Conservatively requires that all dimensions are equal in both
/// tensor products.
fn tensors_are_ortho(bvs1: &[Vector], bvs2: &[Vector]) -> bool {
    if bvs1.len() != bvs2.len() {
        return false; // Malformed, probably
    } else if bvs1.is_empty() {
        return false; // That is even more likely to be malformed
    }

    // First, make sure dimension line up so that our orthogonality check even
    // makes sense. This assumes that there are no nested tensors
    for (bv_1, bv_2) in zip(bvs1, bvs2) {
        match (bv_1.get_explicit_dim(), bv_2.get_explicit_dim()) {
            (Some(dim1), Some(dim2)) if dim1 == dim2 => {
                // keep going
            }
            _ => {
                return false;
            }
        }
    }

    // Now search for some orthogonality

    for (bv_1, bv_2) in zip(bvs1, bvs2) {
        if basis_vectors_are_ortho(bv_1, bv_2) {
            return true;
        }
    }

    // Couldn't find any orthogonality. Conservatively assume there ain't any
    return false;
}

/// Checks the O-SupNeg rule. In other words, verifies the following:
/// ```ignore
/// bv_1a@angle_deg1a + bv_2a@angle_deg2a _|_ bv_1b@angle_deg1b + bv_2b@angle_deg2b
/// ```
fn supneg_ortho(
    bv_1a: &Vector,
    angle_deg_1a: f64,
    bv_2a: &Vector,
    angle_deg_2a: f64,
    bv_1b: &Vector,
    angle_deg_1b: f64,
    bv_2b: &Vector,
    angle_deg_2b: f64,
) -> bool {
    bv_1a == bv_1b
        && bv_2a == bv_2b
        && basis_vectors_are_ortho(bv_1a, bv_2a)
        && (on_phase(angle_deg_1a, angle_deg_2a) && off_phase(angle_deg_1b, angle_deg_2b)
            || off_phase(angle_deg_1a, angle_deg_2a) && on_phase(angle_deg_1b, angle_deg_2b))
}

/// Checks whether `(bv_1a + bv_2a) _|_ (bv_1b + bv_2b)` using the O-Sup
/// and O-SupNeg rules (_not_ any structural rules such as O-Sym). That is,
/// this is expected to be called on the `q1` and `q2` of a
/// `Vector::UniformVectorSuperpos`.
fn superpos_are_ortho(bv_1a: &Vector, bv_2a: &Vector, bv_1b: &Vector, bv_2b: &Vector) -> bool {
    match ((bv_1a, bv_2a), (bv_1b, bv_2b)) {
        (
            (
                Vector::VectorTilt {
                    q: inner_bv_1a,
                    angle_deg: angle_deg_1a,
                    ..
                },
                Vector::VectorTilt {
                    q: inner_bv_2a,
                    angle_deg: angle_deg_2a,
                    ..
                },
            ),
            (
                Vector::VectorTilt {
                    q: inner_bv_1b,
                    angle_deg: angle_deg_1b,
                    ..
                },
                Vector::VectorTilt {
                    q: inner_bv_2b,
                    angle_deg: angle_deg_2b,
                    ..
                },
            ),
        ) if supneg_ortho(
            inner_bv_1a,
            *angle_deg_1a,
            inner_bv_2a,
            *angle_deg_2a,
            inner_bv_1b,
            *angle_deg_1b,
            inner_bv_2b,
            *angle_deg_2b,
        ) =>
        {
            true
        } // O-SupNeg

        (
            (
                _,
                Vector::VectorTilt {
                    q: inner_bv_2a,
                    angle_deg: angle_deg_2a,
                    ..
                },
            ),
            _,
        ) if supneg_ortho(
            bv_1a,
            0.0,
            inner_bv_2a,
            *angle_deg_2a,
            bv_1b,
            0.0,
            bv_2b,
            0.0,
        ) =>
        {
            true
        } // O-SupNeg + O-SupNoTilt

        (
            (
                _,
                Vector::VectorTilt {
                    q: inner_bv_2a,
                    angle_deg: angle_deg_2a,
                    ..
                },
            ),
            (
                Vector::VectorTilt {
                    q: inner_bv_1b,
                    angle_deg: angle_deg_1b,
                    ..
                },
                _,
            ),
        ) if supneg_ortho(
            bv_1a,
            0.0,
            inner_bv_2a,
            *angle_deg_2a,
            inner_bv_1b,
            *angle_deg_1b,
            bv_2b,
            0.0,
        ) =>
        {
            true
        } // O-SupNeg + O-SupNoTilt

        ((inner_bv_1, inner_bv_2), (inner_bv_3, inner_bv_4))
            if basis_vectors_are_ortho(inner_bv_1, inner_bv_2)
                && basis_vectors_are_ortho(inner_bv_1, inner_bv_3)
                && basis_vectors_are_ortho(inner_bv_1, inner_bv_4)
                && basis_vectors_are_ortho(inner_bv_2, inner_bv_3)
                && basis_vectors_are_ortho(inner_bv_2, inner_bv_4)
                && basis_vectors_are_ortho(inner_bv_3, inner_bv_4) =>
        {
            true
        } // O-Sup,

        _ => false,
    }
}

/// Apply the structural rules (O-Sym and O-SupShuf) to attempt
/// `superpos_are_ortho()` with different / orderings of the superpos operands.
fn superpos_are_ortho_sym(
    vec_1a: &Vector,
    vec_2a: &Vector,
    vec_1b: &Vector,
    vec_2b: &Vector,
) -> bool {
    superpos_are_ortho(vec_1a, vec_2a, vec_1b, vec_2b)
        || superpos_are_ortho(vec_1a, vec_2a, vec_2b, vec_1b)
        || superpos_are_ortho(vec_2a, vec_1a, vec_1b, vec_2b)
        || superpos_are_ortho(vec_2a, vec_1a, vec_2b, vec_1b)
}

/// Determine if basis vectors are orthogonal without using O-Sym
fn basis_vectors_are_ortho_nosym(bv_1: &Vector, bv_2: &Vector) -> bool {
    // TODO: need to normalize first, i.e., remove nested tensors
    match (bv_1, bv_2) {
        (Vector::ZeroVector { .. }, Vector::OneVector { .. }) => true, // O-Std

        (Vector::VectorTilt { q: inner_bv_1, .. }, _)
            if basis_vectors_are_ortho(inner_bv_1, bv_2) =>
        {
            true
        } // O-Tilt

        (
            Vector::VectorTensor { qs: inner_bvs1, .. },
            Vector::VectorTensor { qs: inner_bvs2, .. },
        ) if tensors_are_ortho(&inner_bvs1, &inner_bvs2) => true, // O-Tens + O-Shuf

        (
            Vector::UniformVectorSuperpos {
                q1: inner_bv_1a,
                q2: inner_bv_2a,
                ..
            },
            Vector::UniformVectorSuperpos {
                q1: inner_bv_1b,
                q2: inner_bv_2b,
                ..
            },
        ) if superpos_are_ortho_sym(&*inner_bv_1a, &*inner_bv_2a, &*inner_bv_1b, &*inner_bv_2b) => {
            true
        }

        _ => false,
    }
}

/// Determine if two basis vectors are orthogonal using all available
/// orthogonality rules. Practically, this means attempting
/// `basis_vectors_are_ortho()` and then trying again after applying O-Sym.
fn basis_vectors_are_ortho(bv_1: &Vector, bv_2: &Vector) -> bool {
    // O-Sym
    basis_vectors_are_ortho_nosym(bv_1, bv_2) || basis_vectors_are_ortho_nosym(bv_2, bv_1)
}

fn qlits_are_ortho(qlit1: &QLit, qlit2: &QLit) -> bool {
    basis_vectors_are_ortho(
        &qlit1.convert_to_basis_vector(),
        &qlit2.convert_to_basis_vector(),
    )
}

/// Typecheck a QLit node.
/// TODO: Enforce Qwerty rules about QLit types and quantum registers.
fn typecheck_qlit(qlit: &QLit, _env: &mut TypeEnv) -> Result<Type, TypeError> {
    match qlit {
        QLit::ZeroQubit { .. } | QLit::OneQubit { .. } => Ok(Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 1,
        }),

        QLit::QubitTilt { q, .. } => typecheck_qlit(q, _env),

        QLit::UniformSuperpos { q1, q2, .. } => {
            let t1 = typecheck_qlit(q1, _env)?;
            let t2 = typecheck_qlit(q2, _env)?;
            if t1 != t2 {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t1),
                        found: format!("{:?}", t2),
                    },
                    dbg: None,
                })
            } else if !qlits_are_ortho(q1, q2) {
                Err(TypeError {
                    kind: TypeErrorKind::NotOrthogonal {
                        left: format!("{:?}", q1),
                        right: format!("{:?}", q2),
                    },
                    dbg: None,
                })
            } else {
                Ok(t1)
            }
        }

        QLit::QubitTensor { qs, .. } => {
            // TODO: Combine types; for now, just check all are Qubits.
            for q in qs {
                let t = typecheck_qlit(q, _env)?;
                if t != (Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: 1,
                }) {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(format!("{:?}", t)),
                        dbg: None,
                    });
                }
            }
            Ok(Type::RegType {
                elem_ty: RegKind::Qubit,
                // dim: qs.len() as u32,
                dim: qs.len() as u64,
            })
        }
    }
}

/// Typecheck a Vector node (see grammar for rules).
fn typecheck_vector(vector: &Vector, _env: &mut TypeEnv) -> Result<Type, TypeError> {
    match vector {
        Vector::ZeroVector { .. }
        | Vector::OneVector { .. }
        | Vector::PadVector { .. }
        | Vector::TargetVector { .. } => Ok(Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 1,
        }),

        Vector::VectorTilt { q, .. } => typecheck_vector(q, _env),

        Vector::UniformVectorSuperpos { q1, q2, .. } => {
            let t1 = typecheck_vector(q1, _env)?;
            let t2 = typecheck_vector(q2, _env)?;
            if t1 == t2 {
                Ok(t1)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t1),
                        found: format!("{:?}", t2),
                    },
                    dbg: None,
                })
            }
        }

        Vector::VectorTensor { qs, .. } => {
            for q in qs {
                let t = typecheck_vector(q, _env)?;
                if t != (Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: 1,
                }) {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(format!("{:?}", t)),
                        dbg: None,
                    });
                }
            }
            Ok(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: qs.len() as u64,
            })
        }

        Vector::VectorUnit { .. } => Ok(Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 0,
        }),
    }
}

/// Typecheck a Basis node.
/// TODO: Enforce more quantum rules as per Qwerty basis specification.
fn typecheck_basis(basis: &Basis, env: &mut TypeEnv) -> Result<Type, TypeError> {
    match basis {
        Basis::BasisLiteral { vecs, dbg } => {
            if vecs.is_empty() {
                return Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: dbg.clone(),
                });
            }

            let first_ty = typecheck_vector(&vecs[0], env)?;

            if let Type::RegType { elem_ty, dim } = &first_ty {
                // TODO: use dbg of vecs[0], not dbg of basis literal
                if *elem_ty != RegKind::Qubit {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidBasis,
                        dbg: dbg.clone(),
                    });
                }
                if *dim < 1 {
                    return Err(TypeError {
                        kind: TypeErrorKind::EmptyLiteral,
                        dbg: dbg.clone(),
                    });
                }

                vecs.iter().try_for_each(|v| {
                    typecheck_vector(v, env).and_then(|ty| {
                        if ty == first_ty {
                            Ok(())
                        } else {
                            // TODO: use dbg of v, not dbg of basis literal
                            Err(TypeError {
                                kind: TypeErrorKind::DimMismatch,
                                dbg: dbg.clone(),
                            })
                        }
                    })
                })?;

                for (i, v_1) in vecs.iter().enumerate() {
                    for v_2 in &vecs[i + 1..] {
                        if !basis_vectors_are_ortho(v_1, v_2) {
                            return Err(TypeError {
                                kind: TypeErrorKind::NotOrthogonal {
                                    left: v_1.to_programmer_str(),
                                    right: v_2.to_programmer_str(),
                                },
                                dbg: dbg.clone(),
                            });
                        }
                    }
                }

                Ok(Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim: *dim,
                })
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::InvalidBasis,
                    dbg: dbg.clone(),
                })
            }
        }

        Basis::EmptyBasisLiteral { .. } => Ok(Type::RegType {
            elem_ty: RegKind::Basis,
            dim: 0,
        }),

        Basis::BasisTensor { bases, .. } => {
            for b in bases {
                typecheck_basis(b, env)?;
            }
            Ok(Type::UnitType)
        }
    }
}

//
// ─── UNIT TESTS ─────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod basis_tests;
#[cfg(test)]
mod core_tests;
#[cfg(test)]
mod vector_qlit_tests;
