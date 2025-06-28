//! Qwerty typechecker implementation: walks the AST and enforces all typing rules.

use crate::ast::*;
use crate::error::{TypeError, TypeErrorKind};
use crate::span::SourceSpan;
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
    // (If a variable with the same name as lhs already exists in the environment, this will overwrite (shadow) the previous binding with the new type)
    pub fn insert_var(&mut self, name: &str, typ: Type) {
        self.vars.insert(name.to_string(), typ);
    }

    // QWERTY follows Python's variable rules: shadowing is allowed.
    // To disallow shadowing, uncomment the code below and update call sites.

    // Disallow Shadowing (TODO: Seems not required, confirm with Austin)
    /*
    pub fn insert_var(&mut self, name: &str, typ: Type) -> Result<(), TypeError> {
        if self.vars.contains_key(name) {
            return Err(TypeError {
                kind: TypeErrorKind::RedefinedVariable(name.to_string()),
                dbg: None,
            });
        }
        self.vars.insert(name.to_string(), typ);
        Ok(())
    }
    // Update Usage:
    // In typecheck_function and typecheck_stmt:
    // >>    env.insert_var(name, ty.clone())?;
    */

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
        Stmt::Assign { lhs, rhs, dbg } => {
            let rhs_ty = typecheck_expr(rhs, env)?;
            env.insert_var(lhs, rhs_ty); // Shadowing allowed for now.
            Ok(())
        }

        Stmt::UnpackAssign {
            lhs: _,
            rhs,
            dbg: _,
        } => {
            // TODO: Implement tuple/list unpacking logic.
            let _rhs_ty = typecheck_expr(rhs, env)?;
            // Qwerty spec needed: Should rhs_ty be tuple? How to handle arity?
            Ok(())
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

        Expr::BasisTranslation { bin, bout, dbg: _ } => {
            // TODO: Ensure translation is between compatible bases.
            /*
            0) ASK Austin!
            1) Typecheck both bases (already done)
            2) Extract relevant info from each basis (e.g. dimension, type).
            3) Compare the properties we care about (e.g. same dimension, same qubit type).
            4) Return an error if they are not compatible.
            */
            typecheck_basis(bin, env)?;
            typecheck_basis(bout, env)?;
            Ok(Type::UnitType)
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

/// Tolerance for floating point comparison
const ATOL: f64 = 1e-12;

/// Returns true iff the two phases are the same angle (up to a multiple of 360)
fn on_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let modulo = diff % 360.0;
    modulo.abs() < ATOL
}

/// Returns true iff the two phases differ by 180 degrees (up to a multiple of
/// 360)
fn off_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mut modulo = diff % 360.0;
    if modulo < 0.0 {
        modulo = -modulo;
    }
    (modulo - 180.0).abs() < ATOL
}

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
        match (bv_1.get_dim(), bv_2.get_dim()) {
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
    }
}

/// Typecheck a Basis node.
/// TODO: Enforce more quantum rules as per Qwerty basis specification.
fn typecheck_basis(basis: &Basis, env: &mut TypeEnv) -> Result<Type, TypeError> {
    match basis {
        Basis::BasisLiteral { vecs, span } => {
            if vecs.is_empty() {
                return Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    span: span.clone(),
                });
            }

            let first_ty = typecheck_vector(&vecs[0], env)?;

            if let Type::RegType { elem_ty, dim } = &first_ty {
                // TODO: use span of vecs[0], not span of basis literal
                if *elem_ty != RegKind::Qubit {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidBasis,
                        span: span.clone(),
                    });
                }
                if *dim < 1 {
                    return Err(TypeError {
                        kind: TypeErrorKind::EmptyLiteral,
                        span: span.clone(),
                    });
                }

                vecs.iter().try_for_each(|v| {
                    typecheck_vector(v, env).and_then(|ty| {
                        if ty == first_ty {
                            Ok(())
                        } else {
                            // TODO: use span of v, not span of basis literal
                            Err(TypeError {
                                kind: TypeErrorKind::DimMismatch,
                                span: span.clone(),
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
                                span: span.clone(),
                            });
                        }
                    }
                }

                Ok(Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim: *dim,
                }) // TODO: Should this return a Basis type?
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::InvalidBasis,
                    span: span.clone(),
                })
            }
        }

        Basis::EmptyBasisLiteral { .. } => Ok(Type::UnitType),

        Basis::BasisTensor { bases, .. } => {
            for b in bases {
                typecheck_basis(b, env)?;
            }
            Ok(Type::UnitType)
        }
    }
}

//
// ─── UNIT TESTS ────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_typecheck_var_and_assign() {
        let prog = Program {
            funcs: vec![FunctionDef {
                name: "main".into(),
                args: vec![(Type::UnitType, "x".into())],
                ret_type: Type::UnitType,
                body: vec![Stmt::Assign {
                    lhs: "y".into(),
                    rhs: Expr::Variable {
                        name: "x".into(),
                        dbg: None,
                    },
                    dbg: None,
                }],
                dbg: None,
            }],
            dbg: None,
        };
        let result = typecheck_program(&prog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_basis_std() {
        let mut type_env = TypeEnv::new();
        // {'0', '1'} : basis[1] because '0' _|_ '1'
        let ast = Basis::BasisLiteral {
            vecs: vec![
                Vector::ZeroVector { span: None },
                Vector::OneVector { span: None },
            ],
            span: None
        };
        let result = typecheck_basis(&ast, &mut type_env);
        assert_eq!(result, Ok(Type::RegType{elem_ty: RegKind::Basis, dim: 1}));
        assert!(type_env.is_empty());
    }

    #[test]
    fn test_typecheck_basis_not_ortho() {
        let span = SourceSpan {
            file: "skippy.py".to_string(),
            line: 42,
            col: 420
        };
        let mut type_env = TypeEnv::new();
        // {'0', '0'} !: basis[1] because '0' !_|_ '0'
        let ast = Basis::BasisLiteral {
            vecs: vec![
                Vector::ZeroVector { span: None },
                Vector::ZeroVector { span: None },
            ],
            span: Some(span.clone())
        };
        let result = typecheck_basis(&ast, &mut type_env);
        assert_eq!(result, Err(TypeError{kind: TypeErrorKind::NotOrthogonal{left: "'0'".to_string(), right: "'0'".to_string()}, span: Some(span)}));
        assert!(type_env.is_empty());
    }

    #[test]
    fn test_qlits_are_ortho_sym() {
        // Base cases: '0' _|_ '1'
        assert!(qlits_are_ortho(
            &QLit::ZeroQubit { dbg: None },
            &QLit::OneQubit { dbg: None }
        ));
        assert!(qlits_are_ortho(
            &QLit::OneQubit { dbg: None },
            &QLit::ZeroQubit { dbg: None }
        ));
        assert!(!qlits_are_ortho(
            &QLit::ZeroQubit { dbg: None },
            &QLit::ZeroQubit { dbg: None }
        ));

        // '0' and '1' _|_ '0' and -'1'
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::OneQubit { dbg: None }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None
                }),
                dbg: None
            }
        ));
        // '0' and -'1' _|_ '0' and '1'
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::OneQubit { dbg: None }),
                dbg: None
            }
        ));
        // '0' and '1' !_|_ '0' and '1'
        assert!(!qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::OneQubit { dbg: None }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::OneQubit { dbg: None }),
                dbg: None
            }
        ));

        // '0'@45 and '1'@45 _|_ '0'@45 and '1'@225
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 225.0,
                    dbg: None
                }),
                dbg: None
            }
        ));

        // '0' and '1'@0 _|_ '0'@180 and '1'
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 0.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None
                }),
                q2: Box::new(QLit::OneQubit { dbg: None }),
                dbg: None
            }
        ));

        // '0' and '1'@5 !_|_ '0'@180 and '1'
        assert!(!qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 5.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None
                }),
                q2: Box::new(QLit::OneQubit { dbg: None }),
                dbg: None
            }
        ));

        // '0'@45 + '1'@225 _|_ '0'@135 + '1'@135
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 225.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 135.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 135.0,
                    dbg: None
                }),
                dbg: None
            }
        ));
        // '0'@45 + '1'@225 !_|_ '0'@0 + '1'@180
        assert!(!qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 225.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 0.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None
                }),
                dbg: None
            }
        ));
        // '0'@45 + '1'@225 !_|_ '0' + '1'@180
        assert!(!qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 225.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None
                }),
                dbg: None
            }
        ));
        // '0'@45 + '1'@225 !_|_ '0' + '1'@37
        assert!(!qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 225.0,
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::ZeroQubit { dbg: None }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::OneQubit { dbg: None }),
                    angle_deg: 37.0,
                    dbg: None
                }),
                dbg: None
            }
        ));

        // '0'*'1' _|_ '0'*'0'
        assert!(qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }
        ));

        // '0'*'1' !_|_ '0'*'1'
        assert!(!qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            }
        ));

        // '0'*'1' _|_ ('0'*'0')@45
        assert!(qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            },
            &QLit::QubitTilt {
                q: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                angle_deg: 45.0,
                dbg: None
            }
        ));

        // '0'*'0' + '0'*'1' _|_ '1'*'0' + '1'*'1'
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                    dbg: None
                }),
                dbg: None
            }
        ));

        // '0'*'0' + '1'*'1' !_|_ '1'*'0' + '1'*'1'
        assert!(!qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                    dbg: None
                }),
                dbg: None
            }
        ));

        // '0'*'0' + '0'*'1' _|_ '1'*'0' + ('1'*'1')@45
        assert!(qlits_are_ortho(
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                    dbg: None
                }),
                dbg: None
            },
            &QLit::UniformSuperpos {
                q1: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                    dbg: None
                }),
                q2: Box::new(QLit::QubitTilt {
                    q: Box::new(QLit::QubitTensor {
                        qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                        dbg: None
                    }),
                    angle_deg: 45.0,
                    dbg: None
                }),
                dbg: None
            }
        ));

        // ('0'*'0' + '0'*'1') * '0' !_|_ '0'*'0'*'0'
        assert!(!qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::QubitTensor {
                            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                            dbg: None
                        }),
                        q2: Box::new(QLit::QubitTensor {
                            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                            dbg: None
                        }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![
                    QLit::ZeroQubit { dbg: None },
                    QLit::ZeroQubit { dbg: None },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            }
        ));

        // ('0'*'0' + '0'*'1') * '0' !_|_ '0'*'0'
        assert!(!qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::QubitTensor {
                            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                            dbg: None
                        }),
                        q2: Box::new(QLit::QubitTensor {
                            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                            dbg: None
                        }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None }],
                dbg: None
            }
        ));

        // _ * _ !_|_ _ * _
        // \___/      \____/
        //   \           /
        //  empty tensor products
        assert!(!qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![],
                dbg: None
            }
        ));

        // '0'*'0' _|_ ('0'@45)*'1'
        assert!(qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![
                    QLit::QubitTilt {
                        q: Box::new(QLit::ZeroQubit { dbg: None }),
                        angle_deg: 45.0,
                        dbg: None
                    },
                    QLit::OneQubit { dbg: None },
                ],
                dbg: None
            },
        ));

        // ('0'+'1')*'0' _|_ ('0'+-'1')*'0'
        assert!(qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::ZeroQubit { dbg: None }),
                        q2: Box::new(QLit::OneQubit { dbg: None }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::ZeroQubit { dbg: None }),
                        q2: Box::new(QLit::QubitTilt {
                            q: Box::new(QLit::OneQubit { dbg: None }),
                            angle_deg: 180.0,
                            dbg: None
                        }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
        ));

        // (('0'*'0')+'1')*'0' !_|_ ('0'+-'1')*'0'
        assert!(!qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::QubitTensor {
                            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None }],
                            dbg: None
                        }),
                        q2: Box::new(QLit::OneQubit { dbg: None }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::ZeroQubit { dbg: None }),
                        q2: Box::new(QLit::QubitTilt {
                            q: Box::new(QLit::OneQubit { dbg: None }),
                            angle_deg: 180.0,
                            dbg: None
                        }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
        ));

        // ((_ * _)+'1')*'0' !_|_ ('0'+-'1')*'0'
        //  \_____/
        //     \
        //      empty tensor product
        assert!(!qlits_are_ortho(
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::QubitTensor {
                            qs: vec![],
                            dbg: None
                        }),
                        q2: Box::new(QLit::OneQubit { dbg: None }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
            &QLit::QubitTensor {
                qs: vec![
                    QLit::UniformSuperpos {
                        q1: Box::new(QLit::ZeroQubit { dbg: None }),
                        q2: Box::new(QLit::QubitTilt {
                            q: Box::new(QLit::OneQubit { dbg: None }),
                            angle_deg: 180.0,
                            dbg: None
                        }),
                        dbg: None
                    },
                    QLit::ZeroQubit { dbg: None }
                ],
                dbg: None
            },
        ));
    }

    // TODO: Add more tests for all language constructs! In separate test file? Later
}
