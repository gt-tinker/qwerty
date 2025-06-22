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
        Self { vars: HashMap::new() }
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
                span: None,
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
        Stmt::Assign { lhs, rhs, span } => {
            let rhs_ty = typecheck_expr(rhs, env)?;
            env.insert_var(lhs, rhs_ty); // Shadowing allowed for now.
            Ok(())
        }

        Stmt::UnpackAssign { lhs: _, rhs, span: _ } => {
            // TODO: Implement tuple/list unpacking logic.
            let _rhs_ty = typecheck_expr(rhs, env)?;
            // Qwerty spec needed: Should rhs_ty be tuple? How to handle arity?
            Ok(())
        }

        Stmt::Return { val, span } => {
            let val_ty = typecheck_expr(val, env)?;
            if &val_ty != expected_ret_type {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", expected_ret_type),
                        found: format!("{:?}", val_ty),
                    },
                    span: span.clone(),
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
        Expr::Variable { name, span } => {
            env.get_var(name)
                .cloned()
                .ok_or(TypeError {
                    kind: TypeErrorKind::UndefinedVariable(name.clone()),
                    span: span.clone(),
                })
        }

        Expr::UnitLiteral { span: _ } => Ok(Type::UnitType),

        Expr::Adjoint { func, span: _ } => {
            // Adjoint should be a function type (unitary/quantum), not classical.
            let func_ty = typecheck_expr(func, env)?;
            // TODO: Enforce Qwerty adjoint typing rules.
            Ok(func_ty)
        }

        Expr::Pipe { lhs, rhs, span: _ } => {
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
                            span: None,
                        });
                    }
                    Ok((**out_ty).clone())
                }
                _ => Err(TypeError {
                    kind: TypeErrorKind::NotCallable(format!("{:?}", rhs_ty)),
                    span: None,
                }),
            }
        }

        Expr::Measure { basis, span: _ } => {
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

        Expr::Discard { span: _ } => Ok(Type::UnitType),

        // A tensor (often a tensor product) means combining multiple quantum states or registers into a larger, composite system
        Expr::Tensor { vals, span: _ } => {
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
                            span: None,
                        });
                    }
                }
                t = Some(vty);
            }
            Ok(t.unwrap_or(Type::UnitType))
        }

        Expr::BasisTranslation { bin, bout, span: _ } => {
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
            span: _,
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
                    span: None,
                });
            }
            Ok(t_ty)
        }

        Expr::NonUniformSuperpos { pairs, span: _ } => {
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
                            span: None,
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
            span: _,
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
                    span: None,
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

fn qlit_dim(qlit: &QLit) -> Option<usize> {
    match qlit {
        QLit::ZeroQubit{..} | QLit::OneQubit{..} => Some(1),
        QLit::QubitTilt{q: inner_qlit, ..} => qlit_dim(inner_qlit),
        QLit::UniformSuperpos{q1: inner_qlit1, q2: inner_qlit2, ..} =>
            match (qlit_dim(inner_qlit1), qlit_dim(inner_qlit2)) {
                (Some(inner_dim1), Some(inner_dim2))
                    if inner_dim1 == inner_dim2
                    => Some(inner_dim1),
                _ => None,
            }
        QLit::QubitTensor{qs: inner_qlits, ..} =>
            if inner_qlits.len() == 0 {
                None
            } else {
                inner_qlits.iter().try_fold(0, |acc, ql| qlit_dim(ql).map(|dim| acc + dim))
            }
    }
}

fn tensors_are_ortho(qlits1: &[QLit], qlits2: &[QLit]) -> bool {
    if qlits1.len() != qlits2.len() {
       return false // Malformed, probably
    } else if qlits1.is_empty() {
       return false // That is even more likely to be malformed
    }

    // First, make sure dimension line up so that our orthogonality check even
    // makes sense. This assumes that there are no nested tensors
    for (qlit1, qlit2) in zip(qlits1, qlits2) {
        match (qlit_dim(qlit1), qlit_dim(qlit2)) {
            (Some(dim1), Some(dim2)) if dim1 == dim2 => {
                // keep going
            },
            _ => {
                return false;
            },
        }
    }

    // Now search for some orthogonality

    for (qlit1, qlit2) in zip(qlits1, qlits2) {
        if qlits_are_ortho(qlit1, qlit2) {
            return true
        }
    }

    // Couldn't find any orthogonality. Conservatively assume there ain't any
    return false
}

fn on_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let modulo = diff % (2.0 * std::f64::consts::PI);
    modulo.abs() < 1e9
}

fn off_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mut modulo = diff % (2.0 * std::f64::consts::PI);
    if modulo < 0.0 {
        modulo = -modulo;
    }
    (modulo - 180.0).abs() < 1e9
}

// TODO: need to normalize first, i.e., remove nested tensors
fn qlits_are_ortho(qlit1: &QLit, qlit2: &QLit) -> bool {
    match (qlit1, qlit2) {
        (QLit::ZeroQubit{..}, QLit::OneQubit{..}) => true, // O-Std
        (QLit::OneQubit{..}, QLit::ZeroQubit{..}) => true, // O-Std + O-Com
        (QLit::QubitTilt{q: inner_qlit1, ..}, _) => qlits_are_ortho(inner_qlit1, qlit2), // O-Tilt
        (_, QLit::QubitTilt{q: inner_qlit2, ..}) => qlits_are_ortho(qlit1, inner_qlit2), // O-Tilt + O-Com
        (QLit::QubitTensor{qs: inner_qlits1, ..},
         QLit::QubitTensor{qs: inner_qlits2, ..}) => tensors_are_ortho(&inner_qlits1, &inner_qlits2), // O-Tens + O-Shuf
        (QLit::UniformSuperpos{q1: QLit::QubitTilt{q: inner_qlit1a, angle_deg: angle_deg_1a, ..},
                               q2: QLit::QubitTilt{q: inner_qlit2a, angle_deg: angle_deg_2a, ..}, ..},
         QLit::UniformSuperpos{q1: QLit::QubitTilt{q: inner_qlit1b, angle_deg: angle_deg_1b, ..},
                               q2: QLit::QubitTilt{q: inner_qlit2b, angle_deg: angle_deg_2b, ..}, ..})
        if inner_qlit1a == inner_qlit1b
           && inner_qlit2a == inner_qlit2b
           && qlits_are_ortho(inner_qlit1a, inner_qlit2a)
           && (on_phase(angle_deg_1a, angle_deg_1b) && off_phase(angle_deg_2a, angle_deg_2b))
              || (off_phase(angle_deg_1a, angle_deg_1b) && on_phase(angle_deg_2a, angle_deg_2b)) => true, // O-SupNeg
        // TODO: need to handle cases where angle_deg is effectively 0 when there is no outer tilt
        (QLit::UniformSuperpos{q1: inner_qlit1, q2: inner_qlit2, ..},
         QLit::UniformSuperpos{q1: inner_qlit3, q2: inner_qlit4, ..}) => // O-Sup
            qlits_are_ortho(inner_qlit1, inner_qlit2)
            && qlits_are_ortho(inner_qlit1, inner_qlit3)
            && qlits_are_ortho(inner_qlit1, inner_qlit4)
            && qlits_are_ortho(inner_qlit2, inner_qlit3)
            && qlits_are_ortho(inner_qlit2, inner_qlit4)
            && qlits_are_ortho(inner_qlit3, inner_qlit4),
        _ => false,
    }
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
            if t1 == t2 {
                Ok(t1)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t1),
                        found: format!("{:?}", t2),
                    },
                    span: None,
                })
            }
        }

        QLit::QubitTensor { qs, .. } => {
            // TODO: Combine types; for now, just check all are Qubits.
            for q in qs {
                let t = typecheck_qlit(q, _env)?;
                if t != (Type::RegType { elem_ty: RegKind::Qubit, dim: 1 }) {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(format!("{:?}", t)),
                        span: None,
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

        Vector::VectorTilt { q, .. } => typecheck_qlit(q, _env),

        Vector::UniformVectorSuperpos { q1, q2, .. } => {
            let t1 = typecheck_qlit(q1, _env)?;
            let t2 = typecheck_qlit(q2, _env)?;
            if t1 == t2 {
                Ok(t1)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: format!("{:?}", t1),
                        found: format!("{:?}", t2),
                    },
                    span: None,
                })
            }
        }

        Vector::VectorTensor { qs, .. } => {
            for q in qs {
                let t = typecheck_qlit(q, _env)?;
                if t != (Type::RegType { elem_ty: RegKind::Qubit, dim: 1 }) {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(format!("{:?}", t)),
                        span: None,
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
            funcs: vec![
                FunctionDef {
                    name: "main".into(),
                    args: vec![(Type::UnitType, "x".into())],
                    ret_type: Type::UnitType,
                    body: vec![
                        Stmt::Assign {
                            lhs: "y".into(),
                            rhs: Expr::Variable {
                                name: "x".into(),
                                span: None,
                            },
                            span: None,
                        }
                    ],
                    span: None,
                }
            ],
            span: None,
        };
        let result = typecheck_program(&prog);
        assert!(result.is_ok());
    }

    // TODO: Add more tests for all language constructs! In separate test file? Later
}
