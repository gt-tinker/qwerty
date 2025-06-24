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

/// Returns number of qubits represented by a qubit literal (|ql| in the
/// Appendix) or None if the qubit literal is malformed.
fn qlit_dim(qlit: &QLit) -> Option<usize> {
    match qlit {
        QLit::ZeroQubit { .. } | QLit::OneQubit { .. } => Some(1),
        QLit::QubitTilt { q: inner_qlit, .. } => qlit_dim(inner_qlit),
        QLit::UniformSuperpos {
            q1: inner_qlit1,
            q2: inner_qlit2,
            ..
        } => match (qlit_dim(inner_qlit1), qlit_dim(inner_qlit2)) {
            (Some(inner_dim1), Some(inner_dim2)) if inner_dim1 == inner_dim2 => Some(inner_dim1),
            _ => None,
        },
        QLit::QubitTensor {
            qs: inner_qlits, ..
        } => {
            if inner_qlits.len() == 0 {
                None
            } else {
                inner_qlits
                    .iter()
                    .try_fold(0, |acc, ql| qlit_dim(ql).map(|dim| acc + dim))
            }
        }
    }
}

/// Implements the O-Shuf and O-Tens rules by scanning a pair of tensor
/// products elementwise for any pair of orthogonal vectors. If it finds one,
/// that pair is sufficient to conclude that the overall tensor products are
/// orthogonal. Conservatively requires that all dimensions are equal in both
/// tensor products.
fn tensors_are_ortho(qlits1: &[QLit], qlits2: &[QLit]) -> bool {
    if qlits1.len() != qlits2.len() {
        return false; // Malformed, probably
    } else if qlits1.is_empty() {
        return false; // That is even more likely to be malformed
    }

    // First, make sure dimension line up so that our orthogonality check even
    // makes sense. This assumes that there are no nested tensors
    for (qlit1, qlit2) in zip(qlits1, qlits2) {
        match (qlit_dim(qlit1), qlit_dim(qlit2)) {
            (Some(dim1), Some(dim2)) if dim1 == dim2 => {
                // keep going
            }
            _ => {
                return false;
            }
        }
    }

    // Now search for some orthogonality

    for (qlit1, qlit2) in zip(qlits1, qlits2) {
        if qlits_are_ortho_sym(qlit1, qlit2) {
            return true;
        }
    }

    // Couldn't find any orthogonality. Conservatively assume there ain't any
    return false;
}

/// Checks the O-SupNeg rule. In other words, verifies the following:
/// ```ignore
/// qlit1a@angle_deg1a + qlit2a@angle_deg2a _|_ qlit1b@angle_deg1b + qlit2b@angle_deg2b
/// ```
fn supneg_ortho(
    qlit1a: &QLit,
    angle_deg_1a: f64,
    qlit2a: &QLit,
    angle_deg_2a: f64,
    qlit1b: &QLit,
    angle_deg_1b: f64,
    qlit2b: &QLit,
    angle_deg_2b: f64,
) -> bool {
    qlit1a == qlit1b
        && qlit2a == qlit2b
        && qlits_are_ortho_sym(qlit1a, qlit2a)
        && (on_phase(angle_deg_1a, angle_deg_2a) && off_phase(angle_deg_1b, angle_deg_2b)
            || off_phase(angle_deg_1a, angle_deg_2a) && on_phase(angle_deg_1b, angle_deg_2b))
}

/// Checks whether `(qlit1a + qlit2a) _|_ (qlit1b + qlit2b)` using the O-Sup
/// and O-SupNeg rules (_not_ any structural rules such as O-Sym). That is,
/// this is expected to be called on the `q1` and `q2` of a `QLit::Superpos`.
fn superpos_are_ortho(qlit1a: &QLit, qlit2a: &QLit, qlit1b: &QLit, qlit2b: &QLit) -> bool {
    match ((qlit1a, qlit2a), (qlit1b, qlit2b)) {
        (
            (
                QLit::QubitTilt {
                    q: inner_qlit1a,
                    angle_deg: angle_deg_1a,
                    ..
                },
                QLit::QubitTilt {
                    q: inner_qlit2a,
                    angle_deg: angle_deg_2a,
                    ..
                },
            ),
            (
                QLit::QubitTilt {
                    q: inner_qlit1b,
                    angle_deg: angle_deg_1b,
                    ..
                },
                QLit::QubitTilt {
                    q: inner_qlit2b,
                    angle_deg: angle_deg_2b,
                    ..
                },
            ),
        ) if supneg_ortho(
            inner_qlit1a,
            *angle_deg_1a,
            inner_qlit2a,
            *angle_deg_2a,
            inner_qlit1b,
            *angle_deg_1b,
            inner_qlit2b,
            *angle_deg_2b,
        ) =>
        {
            true
        } // O-SupNeg

        (
            (
                _,
                QLit::QubitTilt {
                    q: inner_qlit2a,
                    angle_deg: angle_deg_2a,
                    ..
                },
            ),
            _,
        ) if supneg_ortho(
            qlit1a,
            0.0,
            inner_qlit2a,
            *angle_deg_2a,
            qlit1b,
            0.0,
            qlit2b,
            0.0,
        ) =>
        {
            true
        } // O-SupNeg + O-SupNoTilt

        (
            (
                _,
                QLit::QubitTilt {
                    q: inner_qlit2a,
                    angle_deg: angle_deg_2a,
                    ..
                },
            ),
            (
                QLit::QubitTilt {
                    q: inner_qlit1b,
                    angle_deg: angle_deg_1b,
                    ..
                },
                _,
            ),
        ) if supneg_ortho(
            qlit1a,
            0.0,
            inner_qlit2a,
            *angle_deg_2a,
            inner_qlit1b,
            *angle_deg_1b,
            qlit2b,
            0.0,
        ) =>
        {
            true
        } // O-SupNeg + O-SupNoTilt

        ((inner_qlit1, inner_qlit2), (inner_qlit3, inner_qlit4))
            if qlits_are_ortho_sym(inner_qlit1, inner_qlit2)
                && qlits_are_ortho_sym(inner_qlit1, inner_qlit3)
                && qlits_are_ortho_sym(inner_qlit1, inner_qlit4)
                && qlits_are_ortho_sym(inner_qlit2, inner_qlit3)
                && qlits_are_ortho_sym(inner_qlit2, inner_qlit4)
                && qlits_are_ortho_sym(inner_qlit3, inner_qlit4) =>
        {
            true
        } // O-Sup,

        _ => false,
    }
}

/// Apply the structural rules (O-Sym and O-SupShuf) to attempt
/// `superpos_are_ortho()` with different / orderings of the superpos operands.
fn superpos_are_ortho_comm(qlit1a: &QLit, qlit2a: &QLit, qlit1b: &QLit, qlit2b: &QLit) -> bool {
    superpos_are_ortho(qlit1a, qlit2a, qlit1b, qlit2b)
        || superpos_are_ortho(qlit1a, qlit2a, qlit2b, qlit1b)
        || superpos_are_ortho(qlit2a, qlit1a, qlit1b, qlit2b)
        || superpos_are_ortho(qlit2a, qlit1a, qlit2b, qlit1b)
}

/// Determine if qubit literals are orthogonal without using O-Sym
fn qlits_are_ortho(qlit1: &QLit, qlit2: &QLit) -> bool {
    // TODO: need to normalize first, i.e., remove nested tensors
    match (qlit1, qlit2) {
        (QLit::ZeroQubit { .. }, QLit::OneQubit { .. }) => true, // O-Std

        (QLit::QubitTilt { q: inner_qlit1, .. }, _) if qlits_are_ortho_sym(inner_qlit1, qlit2) => {
            true
        } // O-Tilt

        (
            QLit::QubitTensor {
                qs: inner_qlits1, ..
            },
            QLit::QubitTensor {
                qs: inner_qlits2, ..
            },
        ) if tensors_are_ortho(&inner_qlits1, &inner_qlits2) => true, // O-Tens + O-Shuf

        (
            QLit::UniformSuperpos {
                q1: inner_qlit1a,
                q2: inner_qlit2a,
                ..
            },
            QLit::UniformSuperpos {
                q1: inner_qlit1b,
                q2: inner_qlit2b,
                ..
            },
        ) if superpos_are_ortho_comm(
            &*inner_qlit1a,
            &*inner_qlit2a,
            &*inner_qlit1b,
            &*inner_qlit2b,
        ) =>
        {
            true
        }

        _ => false,
    }
}

/// Try `qlits_are_ortho()` and then try again after applying O-Sym
fn qlits_are_ortho_sym(qlit1: &QLit, qlit2: &QLit) -> bool {
    qlits_are_ortho(qlit1, qlit2) || qlits_are_ortho(qlit2, qlit1) // O-Sym
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
            } else if !qlits_are_ortho_sym(q1, q2) {
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
        | Vector::TargetVector { .. } => Ok(Type::UnitType), // TODO: clarify

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
        Basis::BasisLiteral { vecs, .. } => {
            for v in vecs {
                typecheck_vector(v, env)?;
            }
            Ok(Type::UnitType) // TODO: Should this return a Basis type?
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
    fn test_qlits_are_ortho_comm() {
        // Base cases: '0' _|_ '1'
        assert!(qlits_are_ortho_sym(
            &QLit::ZeroQubit { dbg: None },
            &QLit::OneQubit { dbg: None }
        ));
        assert!(qlits_are_ortho_sym(
            &QLit::OneQubit { dbg: None },
            &QLit::ZeroQubit { dbg: None }
        ));
        assert!(!qlits_are_ortho_sym(
            &QLit::ZeroQubit { dbg: None },
            &QLit::ZeroQubit { dbg: None }
        ));

        // '0' and '1' _|_ '0' and -'1'
        assert!(qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
        assert!(!qlits_are_ortho_sym(
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
