//! Qwerty typechecker implementation: walks the AST and enforces all typing rules.

use crate::dbg::DebugLoc;
use crate::error::{TypeError, TypeErrorKind};
use dashu::base::BitTest;
use std::collections::{HashMap, HashSet};
use std::iter::zip;

use crate::ast::{
    classical::{
        self, BinaryOp, Concat, ModMul, ReduceOp, Repeat, RotateOp, Slice, UnaryOp, UnaryOpKind,
    },
    Assign, BitLiteral, Func, FunctionDef, Program, RegKind, Return, Stmt, StmtExpr, Type,
    UnpackAssign, Variable, angles_are_approx_equal, anti_phase,
    in_phase, qpu,
    qpu::{
        Adjoint, Basis, BasisGenerator, BasisTranslation, Conditional, Discard, EmbedClassical,
        Expr, Measure, NonUniformSuperpos, Pipe, Predicated, QLit, QubitRef, Tensor, UnitLiteral,
        Vector, VectorAtomKind,
    },
};

/// Supplements the type judgment with an additional bit of information:
/// whether statements and expressions involved are reversible or not. Used to
/// check that the body of a reversible function is truly reversible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeKind {
    Rev,
    Irrev,
}

impl ComputeKind {
    /// Returns the compute kind `K` such that `K.join(K') == K'` for any `K'`.
    pub fn identity() -> Self {
        ComputeKind::Rev
    }

    /// If a computation involves this computation kind and `other`, returns
    /// the computation kind of that overall operation.
    pub fn join(self, other: ComputeKind) -> ComputeKind {
        match (self, other) {
            (ComputeKind::Rev, ComputeKind::Rev) => ComputeKind::Rev,
            (_, ComputeKind::Irrev) | (ComputeKind::Irrev, _) => ComputeKind::Irrev,
        }
    }
}

//
// ─── TYPE ENVIRONMENT ───────────────────────────────────────────────────────────
//

/// Tracks variable bindings (and potentially functions, quantum registers, etc.)
#[derive(Debug, Clone)]
pub struct TypeEnv {
    vars: HashMap<String, Type>,
    linear_vars_used: HashSet<String>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            linear_vars_used: HashSet::new(),
        }
    }

    /// Attempts to insert a variable into the type context, throwing an error
    /// if it is already defined.
    pub fn insert_var(
        &mut self,
        name: &str,
        typ: Type,
        dbg: &Option<DebugLoc>,
    ) -> Result<(), TypeError> {
        if let None = self.vars.insert(name.to_string(), typ) {
            Ok(())
        } else {
            Err(TypeError {
                kind: TypeErrorKind::RedefinedVariable(name.to_string()),
                dbg: dbg.clone(),
            })
        }
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

impl Program {
    // TODO: (Future-work!) Change it to Multi/Batch Error reporting Result<(), Vec<TypeError>>
    /// Entry point: checks the whole program.
    /// Returns `Ok(())` if well-typed, or a `TypeError` at the first mistake (Fail fast!!)
    pub fn typecheck(&self) -> Result<(), TypeError> {
        let Program { funcs, .. } = self;
        let mut funcs_available = vec![];

        for func_enum in funcs {
            match func_enum {
                Func::Qpu(func) => {
                    // 'func' => '&FunctionDef<qpu::Expr>'
                    func.typecheck(&funcs_available)?;
                    funcs_available.push((func.name.to_string(), func.get_type()));
                }
                Func::Classical(func) => {
                    // '&FunctionDef<classical::Expr>'
                    func.typecheck(&funcs_available)?;
                    funcs_available.push((func.name.to_string(), func.get_type()));
                }
            }
        }

        Ok(())
    }
}

impl<E: TypeCheckable> FunctionDef<E> {
    /// Returns a new `TypeEnv` for type checking the body of this function.
    pub fn new_type_env(&self, funcs_available: &[(String, Type)]) -> Result<TypeEnv, TypeError> {
        let mut env = TypeEnv::new();

        for (name, ty) in funcs_available {
            env.insert_var(name, ty.clone(), &self.dbg)?;
        }

        // Bind function arguments in environment
        for (ty, name) in &self.args {
            env.insert_var(name, ty.clone(), &self.dbg)?;
        }

        Ok(env)
    }

    /// Returns the expected return type for this function.
    pub fn get_expected_ret_type(&self) -> Option<Type> {
        Some(self.ret_type.clone())
    }

    /// Checks if the compute kind of a statement is valid in this function
    /// definition. (Specifically, if the statement is irreversible yet this
    /// function is reversible)
    pub fn check_stmt_compute_kind(&self, compute_kind: ComputeKind) -> Result<(), TypeError> {
        let FunctionDef {
            name, is_rev, dbg, ..
        } = self;

        if *is_rev && matches!(compute_kind, ComputeKind::Irrev) {
            return Err(TypeError {
                // TODO: say which
                kind: TypeErrorKind::NonReversibleOperationInReversibleFunction(name.to_string()),
                dbg: dbg.clone(),
            });
        } else {
            Ok(())
        }
    }

    /// Verifies that all linear values were used at least once. Should be
    /// called after all statements are type checked.
    pub fn final_linearity_check(&self, env: &TypeEnv) -> Result<(), TypeError> {
        let linear_vars: HashSet<_> = env
            .vars
            .iter()
            .filter_map(|(var, ty)| {
                if ty.is_linear() {
                    Some(var.to_string())
                } else {
                    None
                }
            })
            .collect();
        if let Some(unused_var) = linear_vars
            .difference(&env.linear_vars_used)
            .into_iter()
            .next()
        {
            Err(TypeError {
                kind: TypeErrorKind::LinearVariableUnused(unused_var.to_string()),
                dbg: self.dbg.clone(),
            })
        } else {
            Ok(())
        }
    }

    /// Typechecks a single function and its body, including reversibility
    /// validation. The argument `funcs_available` is used to initialize the
    /// type environment with the names of functions defined earlier.
    /// (Recursion and mutual recursion are banned in Qwerty, so the current
    ///  function is not included in this list.)
    pub fn typecheck(&self, funcs_available: &[(String, Type)]) -> Result<(), TypeError> {
        let mut env = self.new_type_env(funcs_available)?;

        // Single Pass: For each statement, check reversibility BEFORE updating environment
        for stmt in &self.body {
            // Then typecheck the statement and update the environment
            let compute_kind = stmt.typecheck(&mut env, self.get_expected_ret_type())?;
            self.check_stmt_compute_kind(compute_kind)?;
        }

        self.final_linearity_check(&env)
    }
}

//
// ─── STATEMENTS ────────────────────────────────────────────────────────────────
//

impl<E: TypeCheckable> Assign<E> {
    pub fn finish_type_checking(
        &self,
        env: &mut TypeEnv,
        rhs_result: &(Type, ComputeKind),
    ) -> Result<ComputeKind, TypeError> {
        let Assign { lhs, dbg, .. } = self;
        let (rhs_ty, result_compute_kind) = rhs_result;
        env.insert_var(lhs, rhs_ty.clone(), dbg)?;
        Ok(*result_compute_kind)
    }

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<ComputeKind, TypeError> {
        let Assign { rhs, .. } = self;
        let rhs_result = rhs.typecheck(env)?;
        self.finish_type_checking(env, &rhs_result)
    }
}

// UnpackAssign checks:
// 1. RHS must be a register type
// 2. Number of LHS variables must match register dimension
// 3. Each LHS variable gets typed as single-element register of same kind "RegType{ elem_ty, dim: 1 }"
impl<E: TypeCheckable> UnpackAssign<E> {
    pub fn finish_type_checking(
        &self,
        env: &mut TypeEnv,
        rhs_result: &(Type, ComputeKind),
    ) -> Result<ComputeKind, TypeError> {
        let UnpackAssign { lhs, dbg, .. } = self;
        let (rhs_ty, compute_kind) = rhs_result;

        match rhs_ty {
            Type::RegType { elem_ty, dim } => {
                if lhs.len() != *dim {
                    return Err(TypeError {
                        kind: TypeErrorKind::WrongArity {
                            expected: *dim as usize,
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
                        dbg,
                    )?;
                }
                Ok(*compute_kind)
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

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<ComputeKind, TypeError> {
        let UnpackAssign { rhs, .. } = self;
        let rhs_result = rhs.typecheck(env)?;
        self.finish_type_checking(env, &rhs_result)
    }
}

impl<E: TypeCheckable> Return<E> {
    /// Performs some final checks after operand was typechecked.
    pub fn finish_type_checking(
        &self,
        val_result: &(Type, ComputeKind),
        expected_ret_type_opt: Option<Type>,
    ) -> Result<ComputeKind, TypeError> {
        let (val_ty, val_compute_kind) = val_result;
        if let Some(expected_ret_ty) = expected_ret_type_opt {
            if *val_ty != expected_ret_ty {
                Err(TypeError {
                    kind: TypeErrorKind::MismatchedTypes {
                        expected: expected_ret_ty.to_string(),
                        found: val_ty.to_string(),
                    },
                    dbg: self.dbg.clone(),
                })
            } else {
                Ok(*val_compute_kind)
            }
        } else {
            Err(TypeError {
                kind: TypeErrorKind::ReturnOutsideFunction,
                dbg: self.dbg.clone(),
            })
        }
    }

    pub fn typecheck(
        &self,
        env: &mut TypeEnv,
        expected_ret_type: Option<Type>,
    ) -> Result<ComputeKind, TypeError> {
        let Return { val, .. } = self;
        let val_result = val.typecheck(env)?;
        self.finish_type_checking(&val_result, expected_ret_type)
    }
}

impl<E: TypeCheckable> Stmt<E> {
    /// Typecheck a statement.
    /// - env: The current variable/type environment.
    /// - expected_ret_type: Used to check Return statements. If None, we are
    ///   outside a function and returns should nto be allowed.
    pub fn typecheck(
        &self,
        env: &mut TypeEnv,
        expected_ret_type: Option<Type>,
    ) -> Result<ComputeKind, TypeError> {
        match self {
            Stmt::Expr(StmtExpr { expr, .. }) => {
                expr.typecheck(env).map(|(_ty, compute_kind)| compute_kind)
            }
            Stmt::Assign(assign) => assign.typecheck(env),
            Stmt::UnpackAssign(unpack) => unpack.typecheck(env),
            Stmt::Return(ret) => ret.typecheck(env, expected_ret_type),
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
    let merged_input_ty = tensor_product_types(&input_tys, false, dbg)?;
    let merged_output_ty = tensor_product_types(&output_tys, false, dbg)?;
    Ok((merged_input_ty, merged_output_ty))
}

/// Takes the tensor product of a list of types, returning either the new type
/// or a TypeError. The argument allow_func determines whether function types
/// are allowed to be present. (The reason why this argument exists is because
/// this function is called recursively on function types, and we want only to
/// take the tensor product of register types across two function types.)
fn tensor_product_types(
    types: &[Type],
    allow_func: bool,
    dbg: &Option<DebugLoc>,
) -> Result<Type, TypeError> {
    // TODO: this allows e.g. []+id. Is that a good idea?
    let nonunit_tys: Vec<_> = types
        .iter()
        .filter(|ty| !matches!(ty, Type::UnitType { .. }))
        .map(|ty| ty.clone())
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

impl Variable {
    pub fn calc_type(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Variable { name, dbg } = self;
        let (var_ty, compute_kind) = if let Some(var_ty) = env.get_var(name) {
            // Surprisingly, referencing a variable is always reversible.
            // How you use the variable is what determines reversibility.
            Ok((var_ty.clone(), ComputeKind::Rev))
        } else {
            Err(TypeError {
                kind: TypeErrorKind::UndefinedVariable(name.to_string()),
                dbg: dbg.clone(),
            })
        }?;

        if var_ty.is_linear() && !env.linear_vars_used.insert(name.to_string()) {
            Err(TypeError {
                kind: TypeErrorKind::LinearVariableUsedTwice(name.to_string()),
                dbg: dbg.clone(),
            })
        } else {
            Ok((var_ty, compute_kind))
        }
    }

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        self.calc_type(env)
    }
}

impl UnitLiteral {
    pub fn calc_type(&self) -> Result<(Type, ComputeKind), TypeError> {
        Ok((Type::UnitType, ComputeKind::Rev))
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        self.calc_type()
    }
}

impl Adjoint {
    pub fn calc_type(
        &self,
        func_result: &(Type, ComputeKind),
    ) -> Result<(Type, ComputeKind), TypeError> {
        let Adjoint { dbg, .. } = self;
        let (func_ty, compute_kind) = func_result;

        // Must be a reversible function on qubit registers only
        match func_ty {
            Type::RevFuncType { in_out_ty } => match &**in_out_ty {
                Type::RegType { elem_ty, dim } if *elem_ty == RegKind::Qubit && *dim > 0 => {
                    Ok((func_ty.clone(), *compute_kind))
                }
                _ => Err(TypeError {
                    kind: TypeErrorKind::InvalidType(format!(
                        "Adjoint only valid for reversible quantum functions that take qubits, but found: {}",
                        in_out_ty
                    )),
                    dbg: dbg.clone(),
                }),
            },

            Type::FuncType { .. } => {
                // Classical functions cannot have an adjoint
                Err(TypeError {
                    kind: TypeErrorKind::InvalidType(format!(
                        "Cannot take adjoint of irreversible function: {}",
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

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Adjoint { func, .. } = self;
        // Adjoint should be a function type (unitary/quantum), not classical.
        let func_result = func.typecheck(env)?;
        self.calc_type(&func_result)
    }
}

impl Pipe {
    pub fn calc_type(
        &self,
        lhs: &(Type, ComputeKind),
        rhs: &(Type, ComputeKind),
    ) -> Result<(Type, ComputeKind), TypeError> {
        let dbg = &self.dbg;
        let (lhs_ty, lhs_compute_kind) = lhs;
        let (rhs_ty, rhs_compute_kind) = rhs;

        let (ty, my_compute_kind) = match rhs_ty {
            Type::FuncType { in_ty, out_ty } => {
                if &**in_ty != lhs_ty {
                    return Err(TypeError {
                        kind: TypeErrorKind::MismatchedTypes {
                            expected: in_ty.to_string(),
                            found: lhs_ty.to_string(),
                        },
                        dbg: dbg.clone(),
                    });
                }
                Ok(((**out_ty).clone(), ComputeKind::Irrev))
            }

            Type::RevFuncType { in_out_ty } => {
                if &**in_out_ty != lhs_ty {
                    return Err(TypeError {
                        kind: TypeErrorKind::MismatchedTypes {
                            expected: in_out_ty.to_string(),
                            found: lhs_ty.to_string(),
                        },
                        dbg: dbg.clone(),
                    });
                }
                Ok(((**in_out_ty).clone(), ComputeKind::Rev))
            }

            Type::UnitType | Type::RegType { .. } | Type::TupleType { .. } => Err(TypeError {
                kind: TypeErrorKind::NotCallable(rhs_ty.to_string()),
                dbg: dbg.clone(),
            }),
        }?;

        let compute_kind = lhs_compute_kind
            .join(*rhs_compute_kind)
            .join(my_compute_kind);
        Ok((ty, compute_kind))
    }

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Pipe { lhs, rhs, .. } = self;
        // Typing rule: lhs type must match rhs function input type.
        let lhs_result = lhs.typecheck(env)?;
        let rhs_result = rhs.typecheck(env)?;
        self.calc_type(&lhs_result, &rhs_result)
    }
}

impl Measure {
    pub fn calc_type(&self, basis_ty: &Type) -> Result<(Type, ComputeKind), TypeError> {
        // TODO: Should this be elsewhere?
        if !self.basis.fully_spans() {
            return Err(TypeError {
                kind: TypeErrorKind::DoesNotFullySpan,
                dbg: self.basis.get_dbg(),
            });
        }

        let basis_dim = if let Type::RegType {
            elem_ty: RegKind::Basis,
            dim,
        } = basis_ty
        {
            if *dim > 0 {
                Ok(*dim)
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: self.basis.get_dbg(),
                })
            }
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InvalidBasis,
                dbg: self.basis.get_dbg(),
            })
        }?;

        let ty = Type::FuncType {
            in_ty: Box::new(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: basis_dim,
            }),
            out_ty: Box::new(Type::RegType {
                elem_ty: RegKind::Bit,
                dim: basis_dim,
            }),
        };
        // Shockingly, this is reversible. That's because it doesn't do
        // anything until it's used.
        Ok((ty, ComputeKind::Rev))
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        let Measure { basis, .. } = self;
        let basis_ty = basis.typecheck()?;
        self.calc_type(&basis_ty)
    }
}

impl Discard {
    pub fn calc_type(&self) -> Result<(Type, ComputeKind), TypeError> {
        let ty = Type::FuncType {
            in_ty: Box::new(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: 1,
            }),
            out_ty: Box::new(Type::UnitType),
        };
        // This is also strangely reversible. The reason is (like measure) it
        // doesn't do anything until you call it.
        Ok((ty, ComputeKind::Rev))
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        self.calc_type()
    }
}

impl Tensor {
    pub fn calc_type(
        &self,
        val_results: &[(Type, ComputeKind)],
    ) -> Result<(Type, ComputeKind), TypeError> {
        let Tensor { vals, dbg } = self;
        assert_eq!(vals.len(), val_results.len());
        let val_types: Vec<_> = val_results
            .iter()
            .map(|(val_ty, _val_compute_kind)| val_ty)
            .cloned()
            .collect();
        let ty = tensor_product_types(&val_types, true, &dbg)?;
        let compute_kind = val_results
            .iter()
            .map(|(_val_ty, val_compute_kind)| *val_compute_kind)
            .fold(ComputeKind::identity(), |acc, val_compute_kind| {
                acc.join(val_compute_kind)
            });
        Ok((ty, compute_kind))
    }

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Tensor { vals, .. } = self;
        let val_results = vals
            .iter()
            .map(|val| val.typecheck(env))
            .collect::<Result<Vec<_>, TypeError>>()?;
        self.calc_type(&val_results)
    }
}

impl BasisTranslation {
    pub fn calc_type(
        &self,
        bin_ty: &Type,
        bout_ty: &Type,
    ) -> Result<(Type, ComputeKind), TypeError> {
        let BasisTranslation { bin, bout, dbg } = self;
        let result_ty = if let Type::RegType {
            elem_ty: RegKind::Basis,
            dim,
        } = bin_ty
        {
            if *dim == 0 {
                return Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: bin.get_dbg(),
                });
            }

            // Type of this basis translation (pending further checks)
            Type::RevFuncType {
                in_out_ty: Box::new(Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: *dim,
                }),
            }
        } else {
            return Err(TypeError {
                kind: TypeErrorKind::InvalidBasis,
                dbg: bin.get_dbg(),
            });
        };

        if bin_ty != bout_ty {
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

        let pad_indices_in = bin
            .get_atom_indices(VectorAtomKind::PadAtom)
            .ok_or(TypeError {
                kind: TypeErrorKind::MismatchedAtoms {
                    atom_kind: VectorAtomKind::PadAtom,
                },
                dbg: bin.get_dbg(),
            })?;
        let pad_indices_out = bout
            .get_atom_indices(VectorAtomKind::PadAtom)
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
            Ok((result_ty, ComputeKind::Rev))
        }
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        let BasisTranslation { bin, bout, .. } = self;
        let bin_ty = bin.typecheck()?;
        let bout_ty = bout.typecheck()?;
        self.calc_type(&bin_ty, &bout_ty)
    }
}

impl Predicated {
    pub fn calc_type(
        &self,
        then_result: &(Type, ComputeKind),
        else_result: &(Type, ComputeKind),
        pred_ty: &Type,
    ) -> Result<(Type, ComputeKind), TypeError> {
        let Predicated { pred, dbg, .. } = self;
        let (t_ty, t_compute_kind) = then_result;
        let (e_ty, e_compute_kind) = else_result;
        let compute_kind = t_compute_kind.join(*e_compute_kind);

        let pred_dim = if let Type::RegType {
            elem_ty: RegKind::Basis,
            dim,
        } = pred_ty
        {
            if *dim == 0 {
                Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: pred.get_dbg(),
                })
            } else {
                Ok(dim)
            }
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InvalidBasis,
                dbg: pred.get_dbg(),
            })
        }?;
        let num_tgt_atoms = pred
            .get_atom_indices(VectorAtomKind::TargetAtom)
            .ok_or(TypeError {
                kind: TypeErrorKind::MismatchedAtoms {
                    atom_kind: VectorAtomKind::TargetAtom,
                },
                dbg: pred.get_dbg(),
            })?
            .len();
        let num_pad_atoms = pred
            .get_atom_indices(VectorAtomKind::PadAtom)
            .ok_or(TypeError {
                kind: TypeErrorKind::MismatchedAtoms {
                    atom_kind: VectorAtomKind::PadAtom,
                },
                dbg: pred.get_dbg(),
            })?
            .len();
        // Basis cannot be trivial
        if *pred_dim == num_tgt_atoms + num_pad_atoms {
            return Err(TypeError {
                kind: TypeErrorKind::InvalidBasis,
                dbg: pred.get_dbg(),
            });
        }

        // Ensure both operands are reversible functions (Same signature required)
        match (t_ty, e_ty) {
            (
                Type::RevFuncType {
                    in_out_ty: t_in_out,
                },
                Type::RevFuncType {
                    in_out_ty: e_in_out,
                },
            ) => {
                if t_in_out != e_in_out {
                    return Err(TypeError {
                        kind: TypeErrorKind::MismatchedTypes {
                            expected: t_ty.to_string(),
                            found: e_ty.to_string(),
                        },
                        dbg: dbg.clone(),
                    });
                }
                if let Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: reg_dim,
                } = &**t_in_out
                {
                    if *reg_dim == 0 {
                        Err(TypeError {
                            kind: TypeErrorKind::InvalidType(
                                "Cannot predicate function that takes no qubits".to_string(),
                            ),
                            dbg: dbg.clone(),
                        })
                    } else {
                        let ty = Type::RevFuncType {
                            in_out_ty: Box::new(Type::RegType {
                                elem_ty: RegKind::Qubit,
                                dim: *pred_dim,
                            }),
                        };
                        Ok((ty, compute_kind))
                    }
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidType(
                            "Can only predicate functions that take qubits".to_string(),
                        ),
                        dbg: dbg.clone(),
                    })
                }
            }

            (Type::RevFuncType { .. }, _) => Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "Predicated expression requires both operands to be reversible functions, but 'else' branch has type: {}",
                    e_ty
                )),
                dbg: dbg.clone(),
            }),

            (_, Type::RevFuncType { .. }) => Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "Predicated expression requires both operands to be reversible functions, but 'then' branch has type: {}",
                    t_ty
                )),
                dbg: dbg.clone(),
            }),

            (_, _) => Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "Predicated expression requires both operands to be reversible functions, found: then={}, else={}",
                    t_ty, e_ty
                )),
                dbg: dbg.clone(),
            }),
        }
    }

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Predicated {
            then_func,
            else_func,
            pred,
            ..
        } = self;
        let t_result = then_func.typecheck(env)?;
        let e_result = else_func.typecheck(env)?;
        let pred_ty = pred.typecheck()?;
        self.calc_type(&t_result, &e_result, &pred_ty)
    }
}

impl NonUniformSuperpos {
    pub fn calc_type(
        &self,
        term_tys: &[(Type, ComputeKind)],
    ) -> Result<(Type, ComputeKind), TypeError> {
        let NonUniformSuperpos { pairs, dbg } = self;

        if pairs.is_empty() {
            return Err(TypeError {
                kind: TypeErrorKind::EmptyLiteral,
                dbg: dbg.clone(),
            });
        }

        let (first_ty, _first_compute_kind) = &term_tys[0];

        if let Type::RegType {
            elem_ty: RegKind::Qubit,
            dim,
        } = first_ty
        {
            if *dim == 0 {
                return Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: dbg.clone(),
                });
            }
        } else {
            // Should be unreachable
            return Err(TypeError {
                kind: TypeErrorKind::InvalidType("Superposition must contain qubits".to_string()),
                dbg: dbg.clone(),
            });
        }

        if let Some((offending_ty, _offending_compute_kind)) = term_tys
            .iter()
            .find(|(term_ty, _term_compute_kind)| term_ty != first_ty)
        {
            return Err(TypeError {
                kind: TypeErrorKind::MismatchedTypes {
                    expected: first_ty.to_string(),
                    found: offending_ty.to_string(),
                },
                dbg: dbg.clone(),
            });
        }

        let total_prob = pairs
            .iter()
            .map(|(prob, _qlit)| prob)
            .fold(0.0, |prob_acc, prob| prob_acc + prob);
        if !angles_are_approx_equal(total_prob, 1.0) {
            return Err(TypeError {
                kind: TypeErrorKind::ProbabilitiesDoNotSumToOne,
                dbg: dbg.clone(),
            });
        }

        let compute_kind = term_tys.iter().fold(
            ComputeKind::identity(),
            |acc, (_term_ty, term_compute_kind)| acc.join(*term_compute_kind),
        );

        Ok((first_ty.clone(), compute_kind))
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        let NonUniformSuperpos { pairs, .. } = self;
        let term_tys = pairs
            .iter()
            .map(|(_prob, qlit)| qlit.typecheck())
            .collect::<Result<Vec<(Type, ComputeKind)>, TypeError>>()?;
        self.calc_type(&term_tys)
    }
}

/// Holds persistent context needed for incrementally type checking a
/// [`Conditional`] expression.
pub struct ConditionalTypeCtx {
    /// Initially this holds the `TypeEnv.linear_vars_used` before the 'then'
    /// branch, but after the 'then' branch, it holds the
    ///`TypeEnv.linear_vars_used` resulting from the 'then' branch.
    backup_linear_vars_used: HashSet<String>,
}

impl Conditional {
    /// Must run before the 'then' branch is type checked.
    pub fn linearity_check_before_then(&self, env: &TypeEnv) -> ConditionalTypeCtx {
        ConditionalTypeCtx {
            backup_linear_vars_used: env.linear_vars_used.clone(),
        }
    }

    /// Must run after the 'then' branch is type checked but before the 'else'
    /// branch is type checked.
    pub fn linearity_check_after_then_before_else(
        &self,
        env: &mut TypeEnv,
        ctx: &mut ConditionalTypeCtx,
    ) {
        // Restore the linear_vars_used in the TypeEnv from before typechecking
        // the 'then' branch and backup the linear_vars_used from typechecking
        // the 'then' branch.
        std::mem::swap(&mut env.linear_vars_used, &mut ctx.backup_linear_vars_used);
    }

    /// Must run after the 'else' branch is type checked.
    pub fn linearity_check_after_else(
        &self,
        env: &TypeEnv,
        ctx: &ConditionalTypeCtx,
    ) -> Result<(), TypeError> {
        if let Some(mismatch_var) = env
            .linear_vars_used
            .symmetric_difference(&ctx.backup_linear_vars_used)
            .into_iter()
            .next()
        {
            Err(TypeError {
                kind: TypeErrorKind::LinearVariableUseMismatch(mismatch_var.to_string()),
                dbg: self.dbg.clone(),
            })
        } else {
            Ok(())
        }
    }

    pub fn calc_type(
        &self,
        then_result: &(Type, ComputeKind),
        else_result: &(Type, ComputeKind),
        cond_result: &(Type, ComputeKind),
    ) -> Result<(Type, ComputeKind), TypeError> {
        let Conditional { dbg, .. } = self;
        let (t_ty, _t_compute_kind) = then_result;
        let (e_ty, _e_compute_kind) = else_result;
        let (cond_ty, _cond_compute_kind) = cond_result;

        if t_ty != e_ty {
            return Err(TypeError {
                kind: TypeErrorKind::MismatchedTypes {
                    expected: t_ty.to_string(),
                    found: e_ty.to_string(),
                },
                dbg: dbg.clone(),
            });
        }

        if !matches!(
            cond_ty,
            Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 1
            }
        ) {
            return Err(TypeError {
                kind: TypeErrorKind::MismatchedTypes {
                    expected: Type::RegType {
                        elem_ty: RegKind::Bit,
                        dim: 1,
                    }
                    .to_string(),
                    found: cond_ty.to_string(),
                },
                dbg: dbg.clone(),
            });
        }

        // Always consider classical conditionals to be irreversible
        Ok((t_ty.clone(), ComputeKind::Irrev))
    }

    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Conditional {
            then_expr,
            else_expr,
            cond,
            ..
        } = self;
        let mut ctx = self.linearity_check_before_then(env);
        let t_result = then_expr.typecheck(env)?;
        self.linearity_check_after_then_before_else(env, &mut ctx);
        let e_result = else_expr.typecheck(env)?;
        self.linearity_check_after_else(env, &ctx)?;
        let c_result = cond.typecheck(env)?;
        self.calc_type(&t_result, &e_result, &c_result)
    }
}

impl BitLiteral {
    pub fn calc_type(&self) -> Result<(Type, ComputeKind), TypeError> {
        let BitLiteral {
            val: _,
            n_bits,
            dbg,
        } = self;

        if *n_bits == 0 {
            Err(TypeError {
                kind: TypeErrorKind::EmptyLiteral,
                dbg: dbg.clone(),
            })
        } else if n_bits.bit_len() > *n_bits {
            // TODO: use a more descriptive error here
            Err(TypeError {
                kind: TypeErrorKind::DimMismatch,
                dbg: dbg.clone(),
            })
        } else {
            let ty = Type::RegType {
                elem_ty: RegKind::Bit,
                dim: *n_bits,
            };
            Ok((ty, ComputeKind::Rev))
        }
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        self.calc_type()
    }
}

impl EmbedClassical {
    pub fn calc_type(&self, _env: &TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        Ok((
            Type::RevFuncType {
                in_out_ty: Box::new(Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: 3,
                }),
            },
            ComputeKind::Rev,
        ))
    }

    pub fn typecheck(&self, env: &TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        self.calc_type(env)
    }
}

impl QubitRef {
    pub fn calc_type(&self) -> Result<(Type, ComputeKind), TypeError> {
        Err(TypeError {
            kind: TypeErrorKind::InvalidIntermediateComputation,
            dbg: None,
        })
    }

    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        self.calc_type()
    }
}

// ────────────────────────────────────────────────────────────────────────────────────

impl Expr {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        match self {
            Expr::Variable(var) => var.typecheck(env),
            Expr::UnitLiteral(unit_lit) => unit_lit.typecheck(),
            Expr::Adjoint(adj) => adj.typecheck(env),
            Expr::Pipe(pipe) => pipe.typecheck(env),
            Expr::Measure(measure) => measure.typecheck(),
            Expr::Discard(discard) => discard.typecheck(),
            Expr::Tensor(tensor) => tensor.typecheck(env),
            Expr::BasisTranslation(btrans) => btrans.typecheck(),
            Expr::Predicated(pred) => pred.typecheck(env),
            Expr::NonUniformSuperpos(superpos) => superpos.typecheck(),
            Expr::Conditional(cond) => cond.typecheck(env),
            Expr::QLit(qlit) => qlit.typecheck(),
            Expr::BitLiteral(bit_lit) => bit_lit.typecheck(),
            Expr::EmbedClassical(embed) => embed.typecheck(env),
            Expr::QubitRef(qref) => qref.typecheck(),
        }
    }
}

// --- EXPRESSIONS (TRAIT DEFINITION) ---
pub trait TypeCheckable {
    fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError>;
}

// --- EXPRESSIONS (QPU IMPLEMENTATION) ---
impl TypeCheckable for qpu::Expr {
    fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        match self {
            qpu::Expr::Variable(var) => var.typecheck(env),
            qpu::Expr::UnitLiteral(unit_lit) => unit_lit.typecheck(),
            qpu::Expr::EmbedClassical(_) => {
                todo!("EmbedClassical typechecking not implemented yet")
            }
            qpu::Expr::Adjoint(adj) => adj.typecheck(env),
            qpu::Expr::Pipe(pipe) => pipe.typecheck(env),
            qpu::Expr::Measure(measure) => measure.typecheck(),
            qpu::Expr::Discard(discard) => discard.typecheck(),
            qpu::Expr::Tensor(tensor) => tensor.typecheck(env),
            qpu::Expr::BasisTranslation(btrans) => btrans.typecheck(),
            qpu::Expr::Predicated(pred) => pred.typecheck(env),
            qpu::Expr::NonUniformSuperpos(superpos) => superpos.typecheck(),
            qpu::Expr::Conditional(cond) => cond.typecheck(env),
            qpu::Expr::QLit(qlit) => qlit.typecheck(),
            qpu::Expr::BitLiteral(bit_lit) => bit_lit.typecheck(),
            qpu::Expr::QubitRef(qref) => qref.typecheck(),
        }
    }
}

// --- EXPRESSIONS (CLASSICAL IMPLEMENTATION) ---
impl TypeCheckable for classical::Expr {
    fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        match self {
            classical::Expr::Variable(var) => var.typecheck(env),
            classical::Expr::BitLiteral(bit_lit) => bit_lit.typecheck(),
            classical::Expr::Slice(slice) => slice.typecheck(env),
            classical::Expr::UnaryOp(unary_op) => unary_op.typecheck(env),
            classical::Expr::BinaryOp(binary_op) => binary_op.typecheck(env),
            classical::Expr::ReduceOp(reduce_op) => reduce_op.typecheck(env),
            classical::Expr::RotateOp(rotate_op) => rotate_op.typecheck(env),
            classical::Expr::Concat(concat) => concat.typecheck(env),
            classical::Expr::Repeat(repeat) => repeat.typecheck(env),
            classical::Expr::ModMul(mod_mul) => mod_mul.typecheck(env),
        }
    }
}

// --- CLASSICAL EXPRESSION IMPLEMENTATIONS ---

impl Slice {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Slice {
            val,
            lower,
            upper,
            dbg,
        } = self;

        // Typecheck the value being sliced
        let (val_ty, val_compute_kind) = val.typecheck(env)?;

        // Must be a bit register
        if let Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } = val_ty
        {
            // Check bounds
            if *upper > dim {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("[{}..{}] on bit[{}]", lower, upper, dim),
                        ty: "upper bound exceeds register size".to_string(),
                    },
                    dbg: dbg.clone(),
                });
            }

            if *lower >= *upper {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("[{}..{}]", lower, upper),
                        ty: "lower bound must be less than upper bound".to_string(),
                    },
                    dbg: dbg.clone(),
                });
            }

            if *lower >= dim {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("[{}..{}] on bit[{}]", lower, upper, dim),
                        ty: "lower bound exceeds register size".to_string(),
                    },
                    dbg: dbg.clone(),
                });
            }

            let slice_width = upper - lower;
            Ok((
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: slice_width,
                },
                val_compute_kind,
            ))
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "slicing requires bit register, found {}",
                    val_ty
                )),
                dbg: dbg.clone(),
            })
        }
    }
}

impl UnaryOp {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let UnaryOp { kind, val, dbg } = self;

        let (val_ty, val_compute_kind) = val.typecheck(env)?;

        match kind {
            UnaryOpKind::Not => {
                // NOT works on any bit register
                if let Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } = val_ty
                {
                    Ok((
                        Type::RegType {
                            elem_ty: RegKind::Bit,
                            dim,
                        },
                        val_compute_kind,
                    ))
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidType(format!(
                            "bitwise NOT requires bit register, found {}",
                            val_ty
                        )),
                        dbg: dbg.clone(),
                    })
                }
            }
        }
    }
}

impl BinaryOp {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let BinaryOp {
            kind: _,
            left,
            right,
            dbg,
        } = self;

        let (left_ty, left_compute_kind) = left.typecheck(env)?;
        let (right_ty, right_compute_kind) = right.typecheck(env)?;

        // Both operands must be bit registers of same dimension
        match (&left_ty, &right_ty) {
            (
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: left_dim,
                },
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: right_dim,
                },
            ) if left_dim == right_dim => {
                let compute_kind = left_compute_kind.join(right_compute_kind);
                Ok((
                    Type::RegType {
                        elem_ty: RegKind::Bit,
                        dim: *left_dim,
                    },
                    compute_kind,
                ))
            }
            (
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    ..
                },
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    ..
                },
            ) => Err(TypeError {
                kind: TypeErrorKind::MismatchedTypes {
                    expected: format!("bit register with {} bits", left_ty),
                    found: format!("bit register with different size: {}", right_ty),
                },
                dbg: dbg.clone(),
            }),
            _ => Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "binary operations require bit registers, found {} and {}",
                    left_ty, right_ty
                )),
                dbg: dbg.clone(),
            }),
        }
    }
}

impl ReduceOp {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let ReduceOp { kind: _, val, dbg } = self;

        let (val_ty, val_compute_kind) = val.typecheck(env)?;

        if let Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } = val_ty
        {
            if dim == 0 {
                return Err(TypeError {
                    kind: TypeErrorKind::EmptyLiteral,
                    dbg: dbg.clone(),
                });
            }

            // Reduction always produces a single bit
            Ok((
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: 1,
                },
                val_compute_kind,
            ))
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "reduce operations require bit register, found {}",
                    val_ty
                )),
                dbg: dbg.clone(),
            })
        }
    }
}

impl RotateOp {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let RotateOp {
            kind: _,
            val,
            amt,
            dbg,
        } = self;

        let (val_ty, val_compute_kind) = val.typecheck(env)?;
        let (amt_ty, amt_compute_kind) = amt.typecheck(env)?;

        // Value must be bit register, amount must be bit register
        match (&val_ty, &amt_ty) {
            (
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: val_dim,
                },
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: _amt_dim,
                },
            ) => {
                let compute_kind = val_compute_kind.join(amt_compute_kind);
                Ok((
                    Type::RegType {
                        elem_ty: RegKind::Bit,
                        dim: *val_dim,
                    },
                    compute_kind,
                ))
            }
            _ => Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "rotate operations require bit registers, found value: {} and amount: {}",
                    val_ty, amt_ty
                )),
                dbg: dbg.clone(),
            }),
        }
    }
}

impl Concat {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Concat { left, right, dbg } = self;

        let (left_ty, left_compute_kind) = left.typecheck(env)?;
        let (right_ty, right_compute_kind) = right.typecheck(env)?;

        match (&left_ty, &right_ty) {
            (
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: left_dim,
                },
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim: right_dim,
                },
            ) => {
                let total_dim = left_dim + right_dim;
                let compute_kind = left_compute_kind.join(right_compute_kind);
                Ok((
                    Type::RegType {
                        elem_ty: RegKind::Bit,
                        dim: total_dim,
                    },
                    compute_kind,
                ))
            }
            _ => Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "concatenation requires bit registers, found {} and {}",
                    left_ty, right_ty
                )),
                dbg: dbg.clone(),
            }),
        }
    }
}

impl Repeat {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let Repeat { val, amt, dbg } = self;

        let (val_ty, val_compute_kind) = val.typecheck(env)?;

        if let Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } = val_ty
        {
            if *amt == 0 {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("* {}", amt),
                        ty: "repeat amount cannot be 0".to_string(),
                    },
                    dbg: dbg.clone(),
                });
            }

            // Check for potential overflow
            if let Some(total_dim) = dim.checked_mul(*amt) {
                Ok((
                    Type::RegType {
                        elem_ty: RegKind::Bit,
                        dim: total_dim,
                    },
                    val_compute_kind,
                ))
            } else {
                Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("* {}", amt),
                        ty: "dimension overflow".to_string(),
                    },
                    dbg: dbg.clone(),
                })
            }
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "repeat operation requires bit register, found {}",
                    val_ty
                )),
                dbg: dbg.clone(),
            })
        }
    }
}

impl ModMul {
    pub fn typecheck(&self, env: &mut TypeEnv) -> Result<(Type, ComputeKind), TypeError> {
        let ModMul {
            x: _x,
            j: _j,
            y,
            mod_n,
            dbg,
        } = self;

        let (y_ty, y_compute_kind) = y.typecheck(env)?;

        // y must be a bit register
        if let Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } = y_ty
        {
            // Check that mod_n is valid
            if *mod_n <= 1 {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("mod_mul(..., mod_n={})", mod_n),
                        ty: "modulus must be > 1".to_string(),
                    },
                    dbg: dbg.clone(),
                });
            }

            // Additional validation: check that 2^dim > mod_n
            // This ensures the register can represent values in the modular arithmetic domain
            let max_representable = 1u64 << dim.min(63); // Prevent overflow
            if *mod_n as u64 >= max_representable {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidOperation {
                        op: format!("mod_mul(..., mod_n={})", mod_n),
                        ty: format!("modulus {} too large for {}-bit register", mod_n, dim),
                    },
                    dbg: dbg.clone(),
                });
            }

            // Result has same dimension as input y
            // ModMul is reversible in the quantum computing context
            Ok((
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                },
                y_compute_kind,
            ))
        } else {
            Err(TypeError {
                kind: TypeErrorKind::InvalidType(format!(
                    "modular multiplication requires bit register, found {}",
                    y_ty
                )),
                dbg: dbg.clone(),
            })
        }
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
/// ```text
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
        && (in_phase(angle_deg_1a, angle_deg_1b) && anti_phase(angle_deg_2a, angle_deg_2b)
            || anti_phase(angle_deg_1a, angle_deg_1b) && in_phase(angle_deg_2a, angle_deg_2b))
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
                _,
                Vector::VectorTilt {
                    q: inner_bv_2b,
                    angle_deg: angle_deg_2b,
                    ..
                },
            ),
        ) if supneg_ortho(
            bv_1a,
            0.0,
            inner_bv_2a,
            *angle_deg_2a,
            bv_1b,
            0.0,
            inner_bv_2b,
            *angle_deg_2b,
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
/// `superpos_are_ortho()` with different orderings of the superpos operands.
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
    let canon_bv_1 = bv_1.canonicalize();
    let canon_bv_2 = bv_2.canonicalize();
    // O-Sym
    basis_vectors_are_ortho_nosym(&canon_bv_1, &canon_bv_2)
        || basis_vectors_are_ortho_nosym(&canon_bv_2, &canon_bv_1)
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

impl QLit {
    pub fn typecheck(&self) -> Result<(Type, ComputeKind), TypeError> {
        match self {
            QLit::ZeroQubit { .. } | QLit::OneQubit { .. } => {
                let ty = Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: 1,
                };
                Ok((ty, ComputeKind::Rev))
            }

            QLit::QubitTilt { q, .. } => q.typecheck(),

            QLit::UniformSuperpos { q1, q2, dbg } => {
                let (t1, compute_kind1) = q1.typecheck()?;
                let (t2, compute_kind2) = q2.typecheck()?;
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
                    let compute_kind = compute_kind1.join(compute_kind2);
                    Ok((t1, compute_kind))
                }
            }

            QLit::QubitTensor { qs, dbg } => {
                let types = qs
                    .iter()
                    .map(|q| q.typecheck())
                    .collect::<Result<Vec<(Type, ComputeKind)>, TypeError>>()?;
                let (total_dim, compute_kind) = types.iter().try_fold(
                    (0, ComputeKind::identity()),
                    |(dim_acc, compute_kind_acc), (ty, compute_kind)| {
                        if let Type::RegType {
                            elem_ty: RegKind::Qubit,
                            dim,
                        } = ty
                        {
                            Ok((dim_acc + dim, compute_kind_acc.join(*compute_kind)))
                        } else {
                            Err(TypeError {
                                kind: TypeErrorKind::InvalidQubitOperation(ty.to_string()),
                                dbg: dbg.clone(),
                            })
                        }
                    },
                )?;
                let ty = Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: total_dim,
                };
                Ok((ty, compute_kind))
            }

            QLit::QubitUnit { .. } => {
                let ty = Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: 0,
                };
                Ok((ty, ComputeKind::Rev))
            }
        }
    }
}

impl Vector {
    pub fn typecheck(&self) -> Result<Type, TypeError> {
        match self {
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
                    q.typecheck()
                }
            }

            Vector::UniformVectorSuperpos { q1, q2, dbg } => {
                let t1 = q1.typecheck()?;
                let t2 = q2.typecheck()?;
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
                    .map(|q| q.typecheck())
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
}

/// Typecheck a Basis node.
/// TODO: Enforce more quantum rules as per Qwerty basis specification.
impl Basis {
    pub fn typecheck(&self) -> Result<Type, TypeError> {
        match self {
            Basis::BasisLiteral { vecs, dbg } => {
                if vecs.is_empty() {
                    return Err(TypeError {
                        kind: TypeErrorKind::EmptyLiteral,
                        dbg: dbg.clone(),
                    });
                }

                let first_ty = vecs[0].typecheck()?;

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
                        v.typecheck().and_then(|ty| {
                            if ty == first_ty {
                                Ok(())
                            } else {
                                Err(TypeError {
                                    kind: TypeErrorKind::DimMismatch,
                                    dbg: v.get_dbg(),
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
                        dbg: vecs[0].get_dbg(),
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
                    basis.typecheck().and_then(|ty| {
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

            Basis::ApplyBasisGenerator {
                basis,
                generator: generator @ BasisGenerator::Revolve { v1, v2, dbg },
                ..
            } => {
                let basis_ty = basis.typecheck()?;
                let basis_dim = if let Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim,
                } = basis_ty
                {
                    Ok(dim)
                } else {
                    Err(TypeError {
                        kind: TypeErrorKind::InvalidBasis,
                        dbg: basis.get_dbg(),
                    })
                }?;

                if basis_dim == 0 {
                    return Err(TypeError {
                        kind: TypeErrorKind::EmptyLiteral,
                        dbg: basis.get_dbg(),
                    });
                }

                if !basis.fully_spans() {
                    return Err(TypeError {
                        kind: TypeErrorKind::DoesNotFullySpan,
                        dbg: basis.get_dbg(),
                    });
                }

                let v1_ty = v1.typecheck()?;
                let v2_ty = v1.typecheck()?;

                if let Type::RegType { elem_ty, dim } = &v1_ty {
                    if *elem_ty != RegKind::Qubit {
                        return Err(TypeError {
                            kind: TypeErrorKind::InvalidBasis,
                            dbg: v1.get_dbg(),
                        });
                    }
                    if *dim == 0 {
                        return Err(TypeError {
                            kind: TypeErrorKind::EmptyLiteral,
                            dbg: v1.get_dbg(),
                        });
                    }
                    if *dim != 1 {
                        // TODO: use a better error
                        return Err(TypeError {
                            kind: TypeErrorKind::InvalidBasis,
                            dbg: v1.get_dbg(),
                        });
                    }
                } else {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidBasis,
                        dbg: v1.get_dbg(),
                    });
                }

                if v2_ty != v1_ty {
                    return Err(TypeError {
                        kind: TypeErrorKind::DimMismatch,
                        dbg: v2.get_dbg(),
                    });
                }

                if !basis_vectors_are_ortho(v1, v2) {
                    return Err(TypeError {
                        kind: TypeErrorKind::NotOrthogonal {
                            left: v1.to_string(),
                            right: v2.to_string(),
                        },
                        dbg: dbg.clone(),
                    });
                }

                Ok(Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim: basis_dim + generator.get_dim(),
                })
            }
        }
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
