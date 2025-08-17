use crate::{
    error::{LowerError, LowerErrorKind, TypeErrorKind},
    meta::{
        DimExpr, DimVar, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, Progress,
        infer::{DimVarAssignments, FuncDimVarAssignments},
        qpu,
    },
};
use dashu::integer::IBig;
use std::collections::HashMap;

mod expand_classical;
mod expand_qpu;

#[derive(Debug)]
pub enum AliasBinding {
    BasisAlias {
        rhs: qpu::MetaBasis,
    },
    BasisAliasRec {
        base_cases: HashMap<IBig, qpu::MetaBasis>,
        recursive_step: Option<(String, qpu::MetaBasis)>,
    },
}

#[derive(Debug)]
pub enum MacroBinding {
    ExprMacro {
        lhs_pat: qpu::ExprMacroPattern,
        rhs: qpu::MetaExpr,
    },
    BasisMacro {
        lhs_pat: qpu::BasisMacroPattern,
        rhs: qpu::MetaExpr,
    },
    BasisGeneratorMacro {
        lhs_pat: qpu::BasisMacroPattern,
        rhs: qpu::MetaBasisGenerator,
    },
}

#[derive(Debug)]
pub enum DimVarValue {
    Unknown,
    Known(IBig),
}

#[derive(Debug)]
pub struct MacroEnv {
    /// Used for allocating new dimension variables
    func_name: String,
    next_internal_dim_var_id: usize,
    pub aliases: HashMap<String, AliasBinding>,
    pub macros: HashMap<String, MacroBinding>,
    pub dim_vars: HashMap<DimVar, DimVarValue>,
    pub vec_symbols: HashMap<char, qpu::MetaVector>,
}

impl MacroEnv {
    pub fn new(func_name: String) -> MacroEnv {
        MacroEnv {
            func_name,
            next_internal_dim_var_id: 0,
            aliases: HashMap::new(),
            macros: HashMap::new(),
            dim_vars: HashMap::new(),
            vec_symbols: HashMap::new(),
        }
    }

    /// Add a temporary dimension variable that we feel reasonably confident
    /// that inference can take care of.
    pub fn allocate_internal_dim_var(&mut self) -> DimVar {
        loop {
            let var_name = format!("__{}", self.next_internal_dim_var_id);
            self.next_internal_dim_var_id += 1;
            let var = DimVar::FuncVar {
                var_name,
                func_name: self.func_name.clone(),
            };
            if self
                .dim_vars
                .insert(var.clone(), DimVarValue::Unknown)
                .is_none()
            {
                break var;
            }
        }
    }
}

impl DimExpr {
    fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> DimExpr {
        match self {
            DimExpr::DimVar { var, .. } => {
                if *var == dim_var {
                    new_dim_expr
                } else {
                    self.clone()
                }
            }

            DimExpr::DimSum { left, right, dbg } => DimExpr::DimSum {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            DimExpr::DimProd { left, right, dbg } => DimExpr::DimProd {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            DimExpr::DimPow { base, pow, dbg } => DimExpr::DimPow {
                base: Box::new(base.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                pow: Box::new(pow.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            DimExpr::DimNeg { val, dbg } => DimExpr::DimNeg {
                val: Box::new(val.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            DimExpr::DimConst { .. } => self.clone(),
        }
    }

    pub fn expand(&self, env: &MacroEnv) -> Result<(DimExpr, Progress), LowerError> {
        match self {
            DimExpr::DimVar { var, dbg } => {
                if let Some(dim_var_value) = env.dim_vars.get(var) {
                    if let DimVarValue::Known(val) = dim_var_value {
                        Ok((
                            DimExpr::DimConst {
                                val: val.clone(),
                                dbg: dbg.clone(),
                            },
                            Progress::Full,
                        ))
                    } else {
                        Ok((self.clone(), Progress::Partial))
                    }
                } else {
                    Err(LowerError {
                        kind: LowerErrorKind::TypeError {
                            kind: TypeErrorKind::UndefinedVariable(var.to_string()),
                        },
                        dbg: dbg.clone(),
                    })
                }
            }

            DimExpr::DimSum { left, right, dbg } => {
                left.expand(env).and_then(|(expanded_left, left_prog)| {
                    right.expand(env).map(|(expanded_right, right_prog)| {
                        match (&expanded_left, &expanded_right, left_prog.join(right_prog)) {
                            (
                                DimExpr::DimConst { val: left_val, .. },
                                DimExpr::DimConst { val: right_val, .. },
                                prog @ Progress::Full,
                            ) => (
                                DimExpr::DimConst {
                                    val: left_val + right_val,
                                    dbg: dbg.clone(),
                                },
                                prog,
                            ),
                            (_, _, prog) => (
                                DimExpr::DimSum {
                                    left: Box::new(expanded_left),
                                    right: Box::new(expanded_right),
                                    dbg: dbg.clone(),
                                },
                                prog,
                            ),
                        }
                    })
                })
            }

            DimExpr::DimProd { left, right, dbg } => {
                left.expand(env).and_then(|(expanded_left, left_prog)| {
                    right.expand(env).map(|(expanded_right, right_prog)| {
                        match (&expanded_left, &expanded_right, left_prog.join(right_prog)) {
                            (
                                DimExpr::DimConst { val: left_val, .. },
                                DimExpr::DimConst { val: right_val, .. },
                                prog @ Progress::Full,
                            ) => (
                                DimExpr::DimConst {
                                    val: left_val * right_val,
                                    dbg: dbg.clone(),
                                },
                                prog,
                            ),
                            (_, _, prog) => (
                                DimExpr::DimProd {
                                    left: Box::new(expanded_left),
                                    right: Box::new(expanded_right),
                                    dbg: dbg.clone(),
                                },
                                prog,
                            ),
                        }
                    })
                })
            }

            DimExpr::DimPow { base, pow, dbg } => {
                base.expand(env).and_then(|(expanded_base, base_prog)| {
                    pow.expand(env).and_then(|(expanded_pow, pow_prog)| {
                        match (&expanded_base, &expanded_pow, base_prog.join(pow_prog)) {
                            (
                                DimExpr::DimConst { val: base_val, .. },
                                DimExpr::DimConst { .. },
                                prog @ Progress::Full,
                            ) => expanded_pow.extract().map(|pow_int| {
                                (
                                    DimExpr::DimConst {
                                        val: base_val.pow(pow_int),
                                        dbg: dbg.clone(),
                                    },
                                    prog,
                                )
                            }),
                            (_, _, prog) => Ok((
                                DimExpr::DimPow {
                                    base: Box::new(expanded_base),
                                    pow: Box::new(expanded_pow),
                                    dbg: dbg.clone(),
                                },
                                prog,
                            )),
                        }
                    })
                })
            }

            DimExpr::DimNeg { val, dbg } => {
                val.expand(env)
                    .map(|(expanded_val, val_prog)| match (&expanded_val, val_prog) {
                        (DimExpr::DimConst { val: val_val, .. }, Progress::Full) => (
                            DimExpr::DimConst {
                                val: -val_val,
                                dbg: dbg.clone(),
                            },
                            val_prog,
                        ),
                        _ => (
                            DimExpr::DimNeg {
                                val: Box::new(expanded_val),
                                dbg: dbg.clone(),
                            },
                            val_prog,
                        ),
                    })
            }

            DimExpr::DimConst { .. } => Ok((self.clone(), Progress::Full)),
        }
    }
}

impl MetaType {
    pub fn expand(&self, env: &MacroEnv) -> Result<(MetaType, Progress), LowerError> {
        match self {
            MetaType::FuncType { in_ty, out_ty } => {
                in_ty.expand(env).and_then(|(expanded_in_ty, in_ty_prog)| {
                    out_ty.expand(env).map(|(expanded_out_ty, out_ty_prog)| {
                        let expanded_func_ty = MetaType::FuncType {
                            in_ty: Box::new(expanded_in_ty),
                            out_ty: Box::new(expanded_out_ty),
                        };
                        let prog = in_ty_prog.join(out_ty_prog);
                        (expanded_func_ty, prog)
                    })
                })
            }

            MetaType::RevFuncType { in_out_ty } => {
                in_out_ty
                    .expand(env)
                    .map(|(expanded_in_out_ty, in_out_ty_prog)| {
                        let expanded_rev_func_ty = MetaType::RevFuncType {
                            in_out_ty: Box::new(expanded_in_out_ty),
                        };
                        (expanded_rev_func_ty, in_out_ty_prog)
                    })
            }

            MetaType::RegType { elem_ty, dim } => {
                dim.expand(env).map(|(expanded_dim, dim_prog)| {
                    let expanded_reg_ty = MetaType::RegType {
                        elem_ty: *elem_ty,
                        dim: expanded_dim,
                    };
                    (expanded_reg_ty, dim_prog)
                })
            }

            MetaType::TupleType { tys } => tys
                .iter()
                .map(|ty| ty.expand(env))
                .collect::<Result<Vec<_>, LowerError>>()
                .map(|ty_pairs| {
                    let (expanded_tys, progs): (Vec<_>, Vec<_>) = ty_pairs.into_iter().unzip();
                    let expanded_tuple_ty = MetaType::TupleType { tys: expanded_tys };
                    let prog = progs.into_iter().fold(Progress::identity(), Progress::join);
                    (expanded_tuple_ty, prog)
                }),

            MetaType::UnitType => Ok((MetaType::UnitType, Progress::Full)),
        }
    }
}

pub trait Expandable {
    fn expand(&self, env: &mut MacroEnv) -> Result<(Self, Progress), LowerError>
    where
        Self: Sized;
}

impl<S: Expandable> MetaFunctionDef<S> {
    fn expand(
        &self,
        dvs: &FuncDimVarAssignments,
    ) -> Result<(MetaFunctionDef<S>, Progress), LowerError> {
        let MetaFunctionDef {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dim_vars,
            dbg,
        } = self;

        let mut env = MacroEnv::new(name.to_string());

        for (dim_var_name, dim_var_value) in dim_vars.iter().zip(dvs.iter()) {
            let dv_val = if let Some(val) = dim_var_value {
                DimVarValue::Known(val.clone())
            } else {
                DimVarValue::Unknown
            };

            let var = DimVar::FuncVar {
                var_name: dim_var_name.to_string(),
                func_name: name.to_string(),
            };
            if env.dim_vars.insert(var, dv_val).is_some() {
                // Duplicate dimvar
                return Err(LowerError {
                    kind: LowerErrorKind::Malformed,
                    dbg: dbg.clone(),
                });
            }
        }
        let expanded_pairs = body
            .iter()
            .map(|stmt| stmt.expand(&mut env))
            .collect::<Result<Vec<_>, LowerError>>()?;
        let (expanded_stmts, stmt_progs): (Vec<_>, Vec<_>) = expanded_pairs.into_iter().unzip();
        let stmt_prog = stmt_progs
            .into_iter()
            .fold(Progress::identity(), |acc, stmt| acc.join(stmt));

        // Why do this? Because we may have inserted some internal dim vars
        // into the context while expanding statements.
        let new_dim_vars = env
            .dim_vars
            .keys()
            .map(|var| {
                if let DimVar::FuncVar {
                    var_name,
                    func_name,
                } = var
                    && func_name == name
                {
                    var_name.clone()
                } else {
                    // In principle, we could ignore this, but it is better to be
                    // noisy to catch bugs
                    panic!(concat!(
                        "Dimvar in final macro environment is either not a ",
                        "function variable or from a different function"
                    ))
                }
            })
            .collect();
        let (expanded_ret_ty, ret_ty_prog) = if let Some(ret_ty) = ret_type {
            let (ty, prog) = ret_ty.expand(&env)?;
            (Some(ty), prog)
        } else {
            // Trivially fully expanded (but inference will flag it as not
            // fully inferred)
            (None, Progress::Full)
        };
        let arg_pairs = args
            .iter()
            .map(|(arg_type, arg_name)| {
                if let Some(arg_ty) = arg_type {
                    arg_ty.expand(&env).map(|(expanded_arg_ty, arg_ty_prog)| {
                        ((Some(expanded_arg_ty), arg_name.to_string()), arg_ty_prog)
                    })
                } else {
                    // Inference will flag this
                    Ok(((None, arg_name.to_string()), Progress::Full))
                }
            })
            .collect::<Result<Vec<_>, LowerError>>()?;
        let (expanded_args, arg_progs): (Vec<_>, Vec<_>) = arg_pairs.into_iter().unzip();
        let args_prog = arg_progs
            .into_iter()
            .fold(Progress::identity(), Progress::join);

        let expanded_func_def = MetaFunctionDef {
            name: name.to_string(),
            args: expanded_args,
            ret_type: expanded_ret_ty,
            body: expanded_stmts,
            is_rev: *is_rev,
            dim_vars: new_dim_vars,
            dbg: dbg.clone(),
        };
        let progress = stmt_prog.join(ret_ty_prog).join(args_prog);
        Ok((expanded_func_def, progress))
    }
}

impl MetaFunc {
    fn expand(&self, dvs: &FuncDimVarAssignments) -> Result<(MetaFunc, Progress), LowerError> {
        match self {
            MetaFunc::Qpu(qpu_func_def) => qpu_func_def
                .expand(dvs)
                .map(|(expanded_func_def, prog)| (MetaFunc::Qpu(expanded_func_def), prog)),

            MetaFunc::Classical(classical_func_def) => classical_func_def
                .expand(dvs)
                .map(|(expanded_func_def, prog)| (MetaFunc::Classical(expanded_func_def), prog)),
        }
    }
}

impl MetaProgram {
    /// Try to expand as many metaQwerty constructs in this program, returning
    /// a new one.
    pub fn expand(
        &self,
        dv_assign: &DimVarAssignments,
    ) -> Result<(MetaProgram, Progress), LowerError> {
        let MetaProgram { funcs, dbg } = self;
        assert_eq!(funcs.len(), dv_assign.len());
        let funcs_pairs = funcs
            .iter()
            .zip(dv_assign.iter())
            .map(|(func, dvs)| func.expand(dvs))
            .collect::<Result<Vec<(MetaFunc, Progress)>, LowerError>>()?;
        let (expanded_funcs, progresses): (Vec<_>, Vec<_>) = funcs_pairs.into_iter().unzip();
        let progress = progresses
            .into_iter()
            .fold(Progress::identity(), |acc, prog| acc.join(prog));

        Ok((
            MetaProgram {
                funcs: expanded_funcs,
                dbg: dbg.clone(),
            },
            progress,
        ))
    }
}
