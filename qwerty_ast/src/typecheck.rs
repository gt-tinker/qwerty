//! Qwerty typechecker implementation: walks the AST and enforces all typing rules.

use crate::ast::*;
use crate::dbg::DebugLoc;
use crate::error::{TypeError, TypeErrorKind};
use dashu::base::BitTest;
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

/// Typechecks a single function and its body, including reversibility validation.
pub fn typecheck_function(func: &FunctionDef) -> Result<(), TypeError> {
    let mut env = TypeEnv::new();

    // Bind function arguments in environment
    for (ty, name) in &func.args {
        env.insert_var(name, ty.clone());
    }

    // Reconstruct and bind the function type for higher-order use (e.g. Adjoint, pipes)
    let full_func_type = func.get_type();

    // CRITICAL FIX: Bind the full function type into the environment under the function's name.
    // This allows expressions like `Adjoint(my_func)` or `q | my_func` to lookup `my_func`
    // and retrieve its callable type (FuncType/RevFuncType).
    env.insert_var(&func.name, full_func_type.clone());

    let expected_ret_type = Some(func.ret_type.clone());
    let is_annotated_reversible = func.is_rev;

    // Single Pass: For each statement, check reversibility BEFORE updating environment
    for stmt in &func.body {
        // 1. If function is marked reversible, check this statement's reversibility
        //    using the CURRENT environment state (before any updates from this statement)
        if is_annotated_reversible {
            if !check_stmt_reversibility(stmt, &env)? {
                return Err(TypeError {
                    kind: TypeErrorKind::NonReversibleOperationInReversibleFunction(
                        func.name.clone(),
                    ),
                    dbg: func.dbg.clone(),
                });
            }
        }

        // 2. Then typecheck the statement and update the environment
        typecheck_stmt(stmt, &mut env, expected_ret_type.clone())?;
    }

    Ok(())
}

//
// ─── STATEMENTS ────────────────────────────────────────────────────────────────
//

/// Typecheck a statement.
/// - env: The current variable/type environment.
/// - expected_ret_type: Used to check Return statements. If None, we are
///   outside a function and returns should nto be allowed.
pub fn typecheck_stmt(
    stmt: &Stmt,
    env: &mut TypeEnv,
    expected_ret_type: Option<Type>,
) -> Result<(), TypeError> {
    match stmt {
        Stmt::Expr { expr, dbg: _ } => {
            typecheck_expr(expr, env)?;
            Ok(())
        }

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
                    if lhs.len() != dim {
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
                                dim: 1, // TODO: DimExpr check!
                            },
                        );
                    }
                    Ok(())
                }
                _ => Err(TypeError {
                    kind: TypeErrorKind::InvalidType(format!(
                        "Can only unpack from register type, found: {}",
                        rhs_ty
                    )),
                    dbg: dbg.clone(),
                }),
            }
        }

        Stmt::Return { val, dbg } => {
            let val_ty = typecheck_expr(val, env)?;
            // TODO: Comparison needs to handle DimExpr equality
            if let Some(expected_ret_ty) = expected_ret_type {
                if val_ty != expected_ret_ty {
                    Err(TypeError {
                        kind: TypeErrorKind::MismatchedTypes {
                            expected: expected_ret_ty.to_string(),
                            found: val_ty.to_string(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok(())
                }
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::ReturnOutsideFunction,
                    dbg: dbg.clone(),
                })
            }
        }
    }
}

//
// ─── EXPRESSIONS ────────────────────────────────────────────────────────────────
//

/// Takes a list of function types and returns the tensor product of all their
/// inputs and all their outputs. The presence of a register type is an
/// TypeError, and the presence of a unit is a panic.
fn tensor_product_func_ins_outs(
    tail: &[Type],
    dbg: &Option<DebugLoc>,
) -> Result<(Type, Type), TypeError> {
    let input_output_ty_pairs: Vec<_> = tail
        .iter()
        .map(|ty| match ty {
            Type::FuncType {
                in_ty: ty_in_ty,
                out_ty: ty_out_ty,
            } => Ok(((**ty_in_ty).clone(), (**ty_out_ty).clone())),
            Type::RevFuncType {
                in_out_ty: ty_in_out_ty,
            } => Ok(((**ty_in_out_ty).clone(), (**ty_in_out_ty).clone())),
            Type::RegType { .. } => Err(TypeError {
                kind: TypeErrorKind::MismatchedTypes {
                    expected: "a function".to_string(),
                    found: ty.to_string(),
                },
                dbg: dbg.clone(),
            }),
            Type::TupleType { .. } => Err(TypeError {
                kind: TypeErrorKind::InvalidType(
                    "Tuples are not allowed as operands in a function tensor product".to_string(),
                ),
                dbg: dbg.clone(),
            }),

            Type::UnitType => unreachable!("units removed in tensor_product_types()"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (input_tys, output_tys): (Vec<_>, Vec<_>) = input_output_ty_pairs.into_iter().unzip();
    let merged_input_ty = tensor_product_types(input_tys, false, dbg)?;
    let merged_output_ty = tensor_product_types(output_tys, false, dbg)?;
    Ok((merged_input_ty, merged_output_ty))
}

/// Takes the tensor product of a list of types, returning either the new type
/// or a TypeError. The argument allow_func determines whether function types
/// are allowed to be present. (The reason why this argument exists is because
/// this function is called recursively on function types, and we want only to
/// take the tensor product of register types across two function types.)
fn tensor_product_types(
    types: Vec<Type>,
    allow_func: bool,
    dbg: &Option<DebugLoc>,
) -> Result<Type, TypeError> {
    // TODO: this allows e.g. []+id. Is that a good idea?
    let nonunit_tys: Vec<_> = types
        .into_iter()
        .filter(|ty| !matches!(ty, Type::UnitType { .. }))
        .collect();

    match &nonunit_tys[..] {
        [] => Ok(Type::UnitType),
        [head_ty] => Ok((*head_ty).clone()),
        [head, tail @ ..] => match &head {
            Type::RegType { elem_ty, dim } => {
                let total_dim = tail.iter().try_fold(*dim, |dim_acc, ty| match ty {
                    Type::RegType {
                        elem_ty: ty_elem_ty,
                        dim: ty_dim,
                    } if elem_ty == ty_elem_ty => Ok(dim_acc + *ty_dim),
                    Type::RegType { .. } | Type::FuncType { .. } | Type::RevFuncType { .. } => {
                        Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: head.to_string(),
                                found: ty.to_string(),
                            },
                            dbg: dbg.clone(),
                        })
                    }
                    Type::TupleType { .. } => Err(TypeError {
                        kind: TypeErrorKind::InvalidType(
                            "Tuples are not allowed in register folding within tensor products"
                                .to_string(),
                        ),
                        dbg: dbg.clone(),
                    }),
                    Type::UnitType => unreachable!("units removed above"),
                })?;
                Ok(Type::RegType {
                    elem_ty: elem_ty.clone(),
                    dim: total_dim,
                })
            }
            Type::FuncType { .. } if allow_func => {
                let (merged_input_ty, merged_output_ty) =
                    tensor_product_func_ins_outs(&nonunit_tys[..], dbg)?;
                Ok(Type::FuncType {
                    in_ty: Box::new(merged_input_ty),
                    out_ty: Box::new(merged_output_ty),
                })
            }
            Type::RevFuncType { .. } if allow_func => {
                let is_rev = tail.iter().all(|ty| matches!(ty, Type::RevFuncType { .. }));
                let (merged_input_ty, merged_output_ty) =
                    tensor_product_func_ins_outs(&nonunit_tys[..], dbg)?;
                if is_rev {
                    assert_eq!(merged_input_ty, merged_output_ty);
                    Ok(Type::RevFuncType {
                        in_out_ty: Box::new(merged_input_ty),
                    })
                } else {
                    Ok(Type::FuncType {
                        in_ty: Box::new(merged_input_ty),
                        out_ty: Box::new(merged_output_ty),
                    })
                }
            }
            Type::FuncType { .. } | Type::RevFuncType { .. } => Err(TypeError {
                kind: TypeErrorKind::UnsupportedTensorProduct,
                dbg: dbg.clone(),
            }),
            Type::TupleType { .. } => Err(TypeError {
                kind: TypeErrorKind::InvalidType(
                    "Tuple types cannot be used in tensor products".to_string(),
                ),
                dbg: dbg.clone(),
            }),
            Type::UnitType => unreachable!("units removed above"),
        },
    }
}

/// Typecheck an expression and return its type.
pub fn typecheck_expr(expr: &Expr, env: &mut TypeEnv) -> Result<Type, TypeError> {
    match expr {
        Expr::Variable { name, dbg } => env.get_var(name).cloned().ok_or(TypeError {
            kind: TypeErrorKind::UndefinedVariable(name.clone()),
            dbg: dbg.clone(),
        }),

        Expr::UnitLiteral { dbg: _ } => Ok(Type::UnitType),

        Expr::Adjoint { func, dbg } => {
            // Adjoint should be a function type (unitary/quantum), not classical.
            let func_ty = typecheck_expr(func, env)?;

            // Must be a reversible function on qubit registers only
            match func_ty {
                Type::RevFuncType { in_out_ty } => match *in_out_ty {
                    Type::RegType { ref elem_ty, dim } if *elem_ty == RegKind::Qubit && dim > 0 => {
                        Ok(Type::RevFuncType { in_out_ty })
                    }
                    _ => Err(TypeError {
                        kind: TypeErrorKind::InvalidType(format!(
                            "Adjoint only valid for reversible quantum functions of type qubit[m] (m > 0), found: {}",
                            in_out_ty
                        )),
                        dbg: dbg.clone(),
                    }),
                },

                Type::FuncType { .. } => {
                    // Classical functions cannot have an adjoint
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidType(format!(
                            "Cannot take adjoint of non-reversible function: {}",
                            func_ty
                        )),
                        dbg: dbg.clone(),
                    })
                }

                _ => Err(TypeError {
                    kind: TypeErrorKind::NotCallable(format!(
                        "Cannot take adjoint of non-function type: {}",
                        func_ty
                    )),
                    dbg: dbg.clone(),
                }),
            }
        }

        Expr::Pipe { lhs, rhs, dbg } => {
            // Typing rule: lhs type must match rhs function input type.
            let lhs_ty = typecheck_expr(lhs, env)?;
            let rhs_ty = typecheck_expr(rhs, env)?;

            match &rhs_ty {
                Type::FuncType { in_ty, out_ty } => {
                    if **in_ty != lhs_ty {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: in_ty.to_string(),
                                found: lhs_ty.to_string(),
                            },
                            dbg: dbg.clone(),
                        });
                    }
                    Ok((**out_ty).clone())
                }

                Type::RevFuncType { in_out_ty } => {
                    if **in_out_ty != lhs_ty {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: in_out_ty.to_string(),
                                found: lhs_ty.to_string(),
                            },
                            dbg: dbg.clone(),
                        });
                    }
                    Ok((**in_out_ty).clone())
                }

                _ => Err(TypeError {
                    kind: TypeErrorKind::NotCallable(rhs_ty.to_string()),
                    dbg: dbg.clone(),
                }),
            }
        }

        Expr::Measure { basis, dbg: _ } => {
            // Qwerty: measurement returns classical result; basis must be valid.
            let basis_ty = typecheck_basis(basis, env)?; //  is it a legal quantum basis?

            let basis_dim = if let Type::RegType {
                elem_ty: RegKind::Basis,
                dim,
            } = basis_ty
            {
                if dim > 0 {
                    Ok(dim)
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::EmptyLiteral,
                        dbg: match basis {
                            Basis::BasisLiteral { dbg, .. } => dbg.clone(),
                            Basis::EmptyBasisLiteral { dbg, .. } => dbg.clone(),
                            Basis::BasisTensor { dbg, .. } => dbg.clone(),
                        },
                    })
                }
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::InvalidBasis,
                    dbg: match basis {
                        Basis::BasisLiteral { dbg, .. } => dbg.clone(),
                        Basis::EmptyBasisLiteral { dbg, .. } => dbg.clone(),
                        Basis::BasisTensor { dbg, .. } => dbg.clone(),
                    },
                })
            }?;

            Ok(Type::FuncType {
                in_ty: Box::new(Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: basis_dim,
                }),
                out_ty: Box::new(Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: basis_dim,
                }),
            })
        }

        Expr::Discard { dbg: _ } => Ok(Type::UnitType),

        Expr::Tensor { vals, dbg } => {
            let val_tys = vals
                .iter()
                .map(|val| typecheck_expr(val, env))
                .collect::<Result<Vec<Type>, TypeError>>()?;
            tensor_product_types(val_tys, true, &dbg)
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

            for b in [bin, bout] {
                if b.get_atom_indices(VectorAtomKind::TargetAtom)
                    .is_none_or(|indices| !indices.is_empty())
                {
                    return Err(TypeError {
                        kind: TypeErrorKind::MismatchedAtoms {
                            atom_kind: VectorAtomKind::TargetAtom,
                        },
                        dbg: b.get_dbg(),
                    });
                }
            }

            let pad_indices_in =
                bin.get_atom_indices(VectorAtomKind::PadAtom)
                    .ok_or(TypeError {
                        kind: TypeErrorKind::MismatchedAtoms {
                            atom_kind: VectorAtomKind::PadAtom,
                        },
                        dbg: bin.get_dbg(),
                    })?;
            let pad_indices_out =
                bout.get_atom_indices(VectorAtomKind::PadAtom)
                    .ok_or(TypeError {
                        kind: TypeErrorKind::MismatchedAtoms {
                            atom_kind: VectorAtomKind::PadAtom,
                        },
                        dbg: bout.get_dbg(),
                    })?;
            if pad_indices_in != pad_indices_out {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedAtoms {
                        atom_kind: VectorAtomKind::PadAtom,
                    },
                    dbg: dbg.clone(),
                });
            }

            if !basis_span_equiv(bin, bout) {
                Err(TypeError {
                    kind: TypeErrorKind::SpanMismatch,
                    dbg: dbg.clone(),
                })
            } else {
                Ok(result_ty)
            }
        }

        Expr::Predicated {
            then_func,
            else_func,
            pred,
            dbg,
        } => {
            let t_ty = typecheck_expr(then_func, env)?;
            let e_ty = typecheck_expr(else_func, env)?;
            let _pred_ty = typecheck_basis(pred, env)?;

            // Ensure both operands are reversible functions (Same signature required)
            match (&t_ty, &e_ty) {
                (Type::RevFuncType { in_out_ty: t_in_out }, Type::RevFuncType { in_out_ty: e_in_out }) => {
                    if t_in_out != e_in_out {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: t_ty.to_string(),
                                found: e_ty.to_string(),
                            },
                            dbg: dbg.clone(),
                        });
                    }
                    Ok(t_ty)
                }

                (Type::RevFuncType { .. }, _) => {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidType(format!(
                            "Predicated expression requires both operands to be reversible functions, but 'else' branch has type: {}",
                            e_ty
                        )),
                        dbg: dbg.clone(),
                    })
                }

                (_, Type::RevFuncType { .. }) => {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidType(format!(
                            "Predicated expression requires both operands to be reversible functions, but 'then' branch has type: {}",
                            t_ty
                        )),
                        dbg: dbg.clone(),
                    })
                }

                (_, _) => {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidType(format!(
                            "Predicated expression requires both operands to be reversible functions, found: then={}, else={}",
                            t_ty, e_ty
                        )),
                        dbg: dbg.clone(),
                    })
                }
            }
        }

        Expr::NonUniformSuperpos { pairs, dbg } => {
            // Each pair is (weight, QLit). All QLits must have same type.
            let mut qt = None;
            for (_, qlit) in pairs {
                let qlit_ty = typecheck_qlit(qlit, env)?;
                if let Some(prev) = &qt {
                    if &qlit_ty != prev {
                        return Err(TypeError {
                            kind: TypeErrorKind::MismatchedTypes {
                                expected: prev.to_string(),
                                found: qlit_ty.to_string(),
                            },
                            dbg: dbg.clone(),
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
            dbg,
        } => {
            let t_ty = typecheck_expr(then_expr, env)?;
            let e_ty = typecheck_expr(else_expr, env)?;
            let _c_ty = typecheck_expr(cond, env)?;
            if t_ty != e_ty {
                return Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: t_ty.to_string(),
                        found: e_ty.to_string(),
                    },
                    dbg: dbg.clone(),
                });
            }
            Ok(t_ty)
        }

        Expr::QLit { qlit, dbg: _ } => typecheck_qlit(qlit, env),

        Expr::BitLiteral { dim, bits, dbg } => {
            if *dim == 0 {
                Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: dbg.clone(),
                })
            } else if bits.bit_len() > *dim {
                // TODO: use a more descriptive error here
                Err(TypeError {
                    kind: TypeErrorKind::DimMismatch,
                    dbg: dbg.clone(),
                })
            } else {
                Ok(Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: *dim,
                })
            }
        }

        Expr::QubitRef { .. } => Err(TypeError {
            kind: TypeErrorKind::InvalidIntermediateComputation,
            dbg: None,
        }),
    }
}

//
// ─── HELPER METHODS ────────────────────────────────────────────────────────────────
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
    bv_1a.strip_dbg() == bv_1b.strip_dbg()
        && bv_2a.strip_dbg() == bv_2b.strip_dbg()
        && basis_vectors_are_ortho(bv_1a, bv_2a)
        && (in_phase(angle_deg_1a, angle_deg_2a) && anti_phase(angle_deg_1b, angle_deg_2b)
            || anti_phase(angle_deg_1a, angle_deg_2a) && in_phase(angle_deg_1b, angle_deg_2b))
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
    // TODO: need to canonicalize first, i.e., remove nested tensors
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

/// Determines if two basis vectors are orthogonal using all available
/// orthogonality rules. Practically, this means attempting
/// `basis_vectors_are_ortho()` and then trying again after applying O-Sym.
fn basis_vectors_are_ortho(bv_1: &Vector, bv_2: &Vector) -> bool {
    // O-Sym
    basis_vectors_are_ortho_nosym(bv_1, bv_2) || basis_vectors_are_ortho_nosym(bv_2, bv_1)
}

/// Attempts to factors the small basis from the big basis and return the
/// remainder. Based on Algorithm B2 of the CGO '25 paper.
fn factor_basis(small: &Basis, small_dim: usize, big: &Basis, big_dim: usize) -> Option<Basis> {
    assert!(
        big_dim > small_dim,
        concat!(
            "Expected to factor a bigger basis from a smaller basis but ",
            "instead you are asking me to factor a {}-qubit basis from a ",
            "{}-qubit basis, and {} >= {}."
        ),
        small_dim,
        big_dim,
        small_dim,
        big_dim,
    );

    if small.fully_spans() && big.fully_spans() {
        let delta = big_dim - small_dim;
        // Return std[𝛿] as the remainder
        Some(Basis::std(delta, big.get_dbg().clone()))
    } else if small.fully_spans() {
        // big does not fully span and it is a basis literal. We need to try
        // and factor a fully-spanning 𝛿-qubit basis out of it.
        // TODO: implement me
        None
    } else {
        // Neither small nor big fully spans. Both are basis literals. Cross
        // your fingers.
        // TODO: implement me
        None
    }
}

/// Returns true if both bases have the same span. Assumes that both bases
/// individually passed type checking. This is Algorithm B1 in the CGO '25
/// paper.
fn basis_span_equiv(b1: &Basis, b2: &Basis) -> bool {
    let mut b1_stack = b1.make_explicit().canonicalize().normalize().to_stack();
    let mut b2_stack = b2.make_explicit().canonicalize().normalize().to_stack();

    loop {
        match (b1_stack.pop(), b2_stack.pop()) {
            // Done
            (None, None) => return true,

            // Dimension mismatch
            (Some(_), None) | (None, Some(_)) => return false,

            (Some(be1), Some(be2)) => {
                match (be1.get_dim(), be2.get_dim()) {
                    // Malformed basis
                    (None, _) | (_, None) => return false,

                    (Some(be1_dim), Some(be2_dim)) => {
                        if be1_dim == be2_dim {
                            if be1.fully_spans() && be2.fully_spans()
                                || be1.strip_dbg() == be2.strip_dbg()
                            {
                                // Looks good. Keep going
                            } else {
                                // Nothing to factor. Game over
                                return false;
                            }
                        } else if be1_dim < be2_dim {
                            match factor_basis(&be1, be1_dim, &be2, be2_dim) {
                                Some(remainder) => {
                                    b2_stack.push(remainder);
                                }
                                None => {
                                    // Couldn't factor
                                    return false;
                                }
                            }
                        } else {
                            // be1_dim > be2_dim
                            match factor_basis(&be2, be2_dim, &be1, be1_dim) {
                                Some(remainder) => {
                                    b1_stack.push(remainder);
                                }
                                None => {
                                    // Couldn't factor
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Returns true if two qubit literals can be proven to be orthogonal.
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

        QLit::UniformSuperpos { q1, q2, dbg } => {
            let t1 = typecheck_qlit(q1, _env)?;
            let t2 = typecheck_qlit(q2, _env)?;
            if t1 != t2 {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: t1.to_string(),
                        found: t2.to_string(),
                    },
                    dbg: dbg.clone(),
                })
            } else if !qlits_are_ortho(q1, q2) {
                Err(TypeError {
                    kind: TypeErrorKind::NotOrthogonal {
                        left: q1.to_string(),
                        right: q2.to_string(),
                    },
                    dbg: dbg.clone(),
                })
            } else {
                Ok(t1)
            }
        }

        QLit::QubitTensor { qs, dbg } => {
            let types = qs
                .iter()
                .map(|q| typecheck_qlit(q, _env))
                .collect::<Result<Vec<Type>, TypeError>>()?;
            let total_dim = types.iter().try_fold(0, |dim_acc, ty| {
                if let Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } = ty
                {
                    Ok(dim_acc + dim)
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(ty.to_string()),
                        dbg: dbg.clone(),
                    })
                }
            })?;
            Ok(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: total_dim,
            })
        }

        QLit::QubitUnit { .. } => Ok(Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 0,
        }),
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

        Vector::VectorTilt { q, angle_deg, dbg } => {
            if !angle_deg.is_finite() {
                Err(TypeError {
                    kind: TypeErrorKind::InvalidFloat { float: *angle_deg },
                    dbg: dbg.clone(),
                })
            } else {
                typecheck_vector(q, _env)
            }
        }

        Vector::UniformVectorSuperpos { q1, q2, dbg } => {
            let t1 = typecheck_vector(q1, _env)?;
            let t2 = typecheck_vector(q2, _env)?;
            if t1 == t2 {
                Ok(t1)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: t1.to_string(),
                        found: t2.to_string(),
                    },
                    dbg: dbg.clone(),
                })
            }
        }

        Vector::VectorTensor { qs, dbg } => {
            let types = qs
                .iter()
                .map(|q| typecheck_vector(q, _env))
                .collect::<Result<Vec<Type>, TypeError>>()?;
            let total_dim = types.iter().try_fold(0, |dim_acc, ty| {
                if let Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } = ty
                {
                    Ok(dim_acc + dim)
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidQubitOperation(ty.to_string()),
                        dbg: dbg.clone(),
                    })
                }
            })?;
            Ok(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: total_dim,
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
                                    left: v_1.to_string(),
                                    right: v_2.to_string(),
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

        Basis::BasisTensor { bases, .. } => bases
            .iter()
            .try_fold(0, |acc_dim, basis| {
                typecheck_basis(basis, env).and_then(|ty| {
                    if let Type::RegType {
                        elem_ty: RegKind::Basis,
                        dim,
                    } = ty
                    {
                        Ok(acc_dim + dim)
                    } else {
                        Err(TypeError {
                            kind: TypeErrorKind::InvalidBasis,
                            dbg: basis.get_dbg(),
                        })
                    }
                })
            })
            .map(|total_dim| Type::RegType {
                elem_ty: RegKind::Basis,
                dim: total_dim,
            }),
    }
}

/// Checks if a single statement is reversible.
/// Checks each statement before updating the environment
fn check_stmt_reversibility(stmt: &Stmt, env: &TypeEnv) -> Result<bool, TypeError> {
    match stmt {
        Stmt::Expr { expr, .. } => is_expr_inherently_reversible(expr, &mut env.clone()),

        Stmt::Assign { rhs, .. } => is_expr_inherently_reversible(rhs, &mut env.clone()),

        Stmt::UnpackAssign { rhs, .. } => is_expr_inherently_reversible(rhs, &mut env.clone()),

        Stmt::Return { val, .. } => {
            if !is_expr_inherently_reversible(val, &mut env.clone())? {
                return Ok(false);
            }
            // A reversible function must return a quantum value (qubit register).
            let val_ty = typecheck_expr(val, &mut env.clone())?;
            match val_ty {
                Type::RegType {
                    elem_ty: RegKind::Qubit,
                    ..
                } => Ok(true),
                _ => Ok(false),
            }
        }
    }
}

/// Determines if an expression is inherently reversible.
fn is_expr_inherently_reversible(expr: &Expr, env: &mut TypeEnv) -> Result<bool, TypeError> {
    match expr {
        // Execution points that can break reversibility
        Expr::Pipe { lhs, rhs, .. } => {
            let lhs_is_rev = is_expr_inherently_reversible(lhs, env)?;
            let rhs_is_rev = is_expr_inherently_reversible(rhs, env)?;
            if !lhs_is_rev || !rhs_is_rev {
                return Ok(false);
            }

            let rhs_ty = typecheck_expr(rhs, env)?; // Get the actual type being called
            match rhs_ty {
                Type::RevFuncType { .. } => Ok(true),
                Type::FuncType { .. } => Ok(false), // Calling irreversible function breaks reversibility
                _ => Ok(false),                     // Invalid function call
            }
        }

        // Classical control flow breaks reversibility
        Expr::Conditional { .. } => Ok(false),

        // Active expressions that need explicit checking
        Expr::Adjoint { func, .. } => {
            let func_ty = typecheck_expr(func, env)?;
            match func_ty {
                Type::RevFuncType { .. } => Ok(true),
                _ => Ok(false), // Adjoint only works on reversible functions
            }
        }

        Expr::Predicated {
            then_func,
            else_func,
            ..
        } => {
            // Predicated operations are reversible only if both branches are reversible
            let then_is_rev = is_expr_inherently_reversible(then_func, env)?;
            let else_is_rev = is_expr_inherently_reversible(else_func, env)?;
            Ok(then_is_rev && else_is_rev)
        }

        Expr::Variable { name, dbg } => {
            if env.get_var(name).is_some() {
                Ok(true)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::UndefinedVariable(name.clone()),
                    dbg: dbg.clone(),
                })
            }
        }

        // Everything else is reversible by default (actual checks for non-reversibity done during execution call)
        _ => Ok(true),
    }
}

//
// ─── UNIT TESTS ─────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod test_typecheck_basis;
#[cfg(test)]
mod test_typecheck_core;
#[cfg(test)]
mod test_typecheck_vec_qlit;
