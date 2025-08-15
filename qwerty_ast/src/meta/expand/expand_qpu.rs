use crate::{
    ast::{angle_is_approx_zero, usize_try_into_angle},
    error::{LowerError, LowerErrorKind},
    meta::{
        DimExpr, DimVar, Progress,
        expand::{AliasBinding, Expandable, MacroBinding, MacroEnv},
        qpu::{
            BasisMacroPattern, ExprMacroPattern, FloatExpr, MetaBasis, MetaBasisGenerator,
            MetaExpr, MetaStmt, MetaVector, RecDefParam,
        },
    },
};
use std::collections::HashMap;

impl FloatExpr {
    pub fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> FloatExpr {
        match self {
            FloatExpr::FloatDimExpr { expr, dbg } => FloatExpr::FloatDimExpr {
                expr: expr.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                dbg: dbg.clone(),
            },

            FloatExpr::FloatSum { left, right, dbg } => FloatExpr::FloatSum {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr.clone())),
                dbg: dbg.clone(),
            },

            FloatExpr::FloatProd { left, right, dbg } => FloatExpr::FloatProd {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr.clone())),
                dbg: dbg.clone(),
            },

            FloatExpr::FloatDiv { left, right, dbg } => FloatExpr::FloatDiv {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr.clone())),
                dbg: dbg.clone(),
            },

            FloatExpr::FloatNeg { val, dbg } => FloatExpr::FloatNeg {
                val: Box::new(val.substitute_dim_var(dim_var, new_dim_expr.clone())),
                dbg: dbg.clone(),
            },

            FloatExpr::FloatConst { .. } => self.clone(),
        }
    }

    pub fn expand(&self, env: &MacroEnv) -> Result<(FloatExpr, Progress), LowerError> {
        match self {
            FloatExpr::FloatDimExpr { expr, dbg } => {
                expr.expand(env).and_then(|(expanded_expr, expr_prog)| {
                    match (&expanded_expr, expr_prog) {
                        (
                            DimExpr::DimConst {
                                val: expr_val,
                                dbg: expr_dbg,
                            },
                            Progress::Full,
                        ) => expanded_expr.extract().and_then(|expr_int| {
                            usize_try_into_angle(expr_int)
                                .ok_or_else(|| LowerError {
                                    kind: LowerErrorKind::IntegerTooBig {
                                        offender: expr_val.clone(),
                                    },
                                    dbg: expr_dbg.clone(),
                                })
                                .map(|expr_float| {
                                    (
                                        FloatExpr::FloatConst {
                                            val: expr_float,
                                            dbg: dbg.clone(),
                                        },
                                        Progress::Full,
                                    )
                                })
                        }),

                        _ => Ok((
                            FloatExpr::FloatDimExpr {
                                expr: expanded_expr,
                                dbg: dbg.clone(),
                            },
                            expr_prog,
                        )),
                    }
                })
            }

            FloatExpr::FloatSum { left, right, dbg } => {
                left.expand(env).and_then(|(expanded_left, left_prog)| {
                    right.expand(env).map(|(expanded_right, right_prog)| {
                        match (&expanded_left, &expanded_right, left_prog.join(right_prog)) {
                            (
                                FloatExpr::FloatConst { val: left_val, .. },
                                FloatExpr::FloatConst { val: right_val, .. },
                                prog @ Progress::Full,
                            ) => (
                                FloatExpr::FloatConst {
                                    val: left_val + right_val,
                                    dbg: dbg.clone(),
                                },
                                prog,
                            ),
                            (_, _, prog) => (
                                FloatExpr::FloatSum {
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

            FloatExpr::FloatProd { left, right, dbg } => {
                left.expand(env).and_then(|(expanded_left, left_prog)| {
                    right.expand(env).map(|(expanded_right, right_prog)| {
                        match (&expanded_left, &expanded_right, left_prog.join(right_prog)) {
                            (
                                FloatExpr::FloatConst { val: left_val, .. },
                                FloatExpr::FloatConst { val: right_val, .. },
                                prog @ Progress::Full,
                            ) => (
                                FloatExpr::FloatConst {
                                    val: left_val * right_val,
                                    dbg: dbg.clone(),
                                },
                                prog,
                            ),
                            (_, _, prog) => (
                                FloatExpr::FloatProd {
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

            FloatExpr::FloatDiv { left, right, dbg } => {
                left.expand(env).and_then(|(expanded_left, left_prog)| {
                    right.expand(env).and_then(|(expanded_right, right_prog)| {
                        match (&expanded_left, &expanded_right, left_prog.join(right_prog)) {
                            (
                                FloatExpr::FloatConst { val: left_val, .. },
                                FloatExpr::FloatConst { val: right_val, .. },
                                prog @ Progress::Full,
                            ) => {
                                if angle_is_approx_zero(*right_val) {
                                    Err(LowerError {
                                        kind: LowerErrorKind::DivisionByZero,
                                        dbg: dbg.clone(),
                                    })
                                } else {
                                    Ok((
                                        FloatExpr::FloatConst {
                                            val: left_val / right_val,
                                            dbg: dbg.clone(),
                                        },
                                        prog,
                                    ))
                                }
                            }
                            (_, _, prog) => Ok((
                                FloatExpr::FloatDiv {
                                    left: Box::new(expanded_left),
                                    right: Box::new(expanded_right),
                                    dbg: dbg.clone(),
                                },
                                prog,
                            )),
                        }
                    })
                })
            }

            FloatExpr::FloatNeg { val, dbg } => {
                val.expand(env)
                    .map(|(expanded_val, val_prog)| match (&expanded_val, val_prog) {
                        (FloatExpr::FloatConst { val: val_val, .. }, Progress::Full) => (
                            FloatExpr::FloatConst {
                                val: -val_val,
                                dbg: dbg.clone(),
                            },
                            val_prog,
                        ),
                        _ => (
                            FloatExpr::FloatNeg {
                                val: Box::new(expanded_val),
                                dbg: dbg.clone(),
                            },
                            val_prog,
                        ),
                    })
            }

            FloatExpr::FloatConst { .. } => Ok((self.clone(), Progress::Full)),
        }
    }
}

impl MetaVector {
    pub fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: MetaVector,
    ) -> MetaVector {
        match self {
            MetaVector::VectorAlias { name, .. } => {
                if *name == vector_alias {
                    new_vector
                } else {
                    self.clone()
                }
            }

            MetaVector::VectorBroadcastTensor { val, factor, dbg } => {
                MetaVector::VectorBroadcastTensor {
                    val: Box::new(val.substitute_vector_alias(vector_alias, new_vector)),
                    factor: factor.clone(),
                    dbg: dbg.clone(),
                }
            }

            MetaVector::VectorTilt { q, angle_deg, dbg } => MetaVector::VectorTilt {
                q: Box::new(q.substitute_vector_alias(vector_alias, new_vector)),
                angle_deg: angle_deg.clone(),
                dbg: dbg.clone(),
            },

            MetaVector::UniformVectorSuperpos { q1, q2, dbg } => {
                MetaVector::UniformVectorSuperpos {
                    q1: Box::new(
                        q1.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                    ),
                    q2: Box::new(q2.substitute_vector_alias(vector_alias, new_vector)),
                    dbg: dbg.clone(),
                }
            }

            MetaVector::VectorBiTensor { left, right, dbg } => MetaVector::VectorBiTensor {
                left: Box::new(
                    left.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                right: Box::new(right.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaVector::VectorSymbol { .. }
            | MetaVector::ZeroVector { .. }
            | MetaVector::OneVector { .. }
            | MetaVector::PadVector { .. }
            | MetaVector::TargetVector { .. }
            | MetaVector::VectorUnit { .. } => self.clone(),
        }
    }

    pub fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> MetaVector {
        match self {
            MetaVector::VectorBroadcastTensor { val, factor, dbg } => {
                MetaVector::VectorBroadcastTensor {
                    val: Box::new(val.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                    factor: factor.substitute_dim_var(dim_var, new_dim_expr.clone()),
                    dbg: dbg.clone(),
                }
            }

            MetaVector::VectorTilt { q, angle_deg, dbg } => MetaVector::VectorTilt {
                q: Box::new(q.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                angle_deg: angle_deg.substitute_dim_var(dim_var, new_dim_expr.clone()),
                dbg: dbg.clone(),
            },

            MetaVector::UniformVectorSuperpos { q1, q2, dbg } => {
                MetaVector::UniformVectorSuperpos {
                    q1: Box::new(q1.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                    q2: Box::new(q2.substitute_dim_var(dim_var, new_dim_expr.clone())),
                    dbg: dbg.clone(),
                }
            }

            MetaVector::VectorBiTensor { left, right, dbg } => MetaVector::VectorBiTensor {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr.clone())),
                dbg: dbg.clone(),
            },

            MetaVector::VectorAlias { .. }
            | MetaVector::VectorSymbol { .. }
            | MetaVector::ZeroVector { .. }
            | MetaVector::OneVector { .. }
            | MetaVector::PadVector { .. }
            | MetaVector::TargetVector { .. }
            | MetaVector::VectorUnit { .. } => self.clone(),
        }
    }

    pub fn expand(&self, env: &MacroEnv) -> Result<(MetaVector, Progress), LowerError> {
        match self {
            // Only substitution can remove this
            MetaVector::VectorAlias { .. } => Ok((self.clone(), Progress::Partial)),

            MetaVector::VectorSymbol { sym, dbg } => if let Some(vec) = env.vec_symbols.get(sym) {
                vec.expand(env)
            } else {
                Err(LowerError {
                    kind: LowerErrorKind::Malformed,
                    dbg: dbg.clone(),
                })
            },

            MetaVector::VectorBroadcastTensor { val, factor, dbg } => val
                .expand(env)
                .and_then(|(expanded_val, val_prog)|
                    factor.expand(env).and_then(|(expanded_factor, factor_prog)|
                        match (&expanded_factor, factor_prog) {
                            (DimExpr::DimConst { .. }, Progress::Full) =>
                                expanded_factor.extract().map(|factor_int|
                                    if factor_int == 0 {
                                        (MetaVector::VectorUnit { dbg: dbg.clone() }, Progress::Full)
                                    } else {
                                        let n_fold_tensor_product = std::iter::repeat(expanded_val)
                                            .take(factor_int)
                                            .reduce(|acc, cloned_val| MetaVector::VectorBiTensor {
                                                left: Box::new(acc),
                                                right: Box::new(cloned_val),
                                                dbg: dbg.clone()
                                            })
                                            .expect("factor_int > 0, so tensor product should not be empty");
                                        (n_fold_tensor_product, val_prog.join(factor_prog))
                                    }
                                ),

                            _ => Ok((
                                MetaVector::VectorBroadcastTensor {
                                    val: Box::new(expanded_val),
                                    factor: expanded_factor,
                                    dbg: dbg.clone()
                                },
                                val_prog.join(factor_prog)
                            )),
                        })),

            MetaVector::VectorTilt { q, angle_deg, dbg } => q
                .expand(env)
                .and_then(|(expanded_q, q_prog)|
                    angle_deg
                        .expand(env)
                        .map(|(expanded_angle, angle_prog)|
                            (MetaVector::VectorTilt {
                                q: Box::new(expanded_q),
                                angle_deg: expanded_angle,
                                dbg: dbg.clone()
                            },
                            q_prog.join(angle_prog)))),

            MetaVector::UniformVectorSuperpos { q1, q2, dbg } => q1
                .expand(env)
                .and_then(|(expanded_q1, q1_prog)|
                    q2.expand(env)
                        .map(|(expanded_q2, q2_prog)|
                            (MetaVector::UniformVectorSuperpos {
                                q1: Box::new(expanded_q1),
                                q2: Box::new(expanded_q2),
                                dbg: dbg.clone(),
                            },
                            q1_prog.join(q2_prog)))),

            MetaVector::VectorBiTensor { left, right, dbg } => left
                .expand(env)
                .and_then(|(expanded_left, left_prog)|
                    right.expand(env)
                        .map(|(expanded_right, right_prog)|
                            (MetaVector::VectorBiTensor {
                                left: Box::new(expanded_left),
                                right: Box::new(expanded_right),
                                dbg: dbg.clone(),
                            },
                            left_prog.join(right_prog)))),

            MetaVector::ZeroVector { .. }
            | MetaVector::OneVector { .. }
            | MetaVector::PadVector { .. }
            | MetaVector::TargetVector { .. }
            | MetaVector::VectorUnit { .. } => Ok((self.clone(), Progress::Full))
        }
    }
}

impl MetaBasisGenerator {
    pub fn substitute_basis_alias(
        &self,
        basis_alias: String,
        new_basis: MetaBasis,
    ) -> MetaBasisGenerator {
        match self {
            MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                MetaBasisGenerator::BasisGeneratorMacro {
                    name: name.to_string(),
                    arg: Box::new(arg.substitute_basis_alias(basis_alias, new_basis)),
                    dbg: dbg.clone(),
                }
            }

            MetaBasisGenerator::Revolve { .. } => self.clone(),
        }
    }

    pub fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: MetaVector,
    ) -> MetaBasisGenerator {
        match self {
            MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                MetaBasisGenerator::BasisGeneratorMacro {
                    name: name.to_string(),
                    arg: Box::new(arg.substitute_vector_alias(vector_alias, new_vector)),
                    dbg: dbg.clone(),
                }
            }

            MetaBasisGenerator::Revolve { v1, v2, dbg } => MetaBasisGenerator::Revolve {
                v1: v1.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                v2: v2.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },
        }
    }

    pub fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> MetaBasisGenerator {
        match self {
            MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                MetaBasisGenerator::BasisGeneratorMacro {
                    name: name.to_string(),
                    arg: Box::new(arg.substitute_dim_var(dim_var, new_dim_expr)),
                    dbg: dbg.clone(),
                }
            }

            MetaBasisGenerator::Revolve { v1, v2, dbg } => MetaBasisGenerator::Revolve {
                v1: v1.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                v2: v2.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
        }
    }

    pub fn expand(&self, env: &MacroEnv) -> Result<(MetaBasisGenerator, Progress), LowerError> {
        match self {
            MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                if let Some(MacroBinding::BasisGeneratorMacro { lhs_pat, rhs }) =
                    env.macros.get(name)
                {
                    arg.expand(env)
                        .and_then(|(expanded_arg, arg_prog)| match lhs_pat {
                            BasisMacroPattern::AnyBasis {
                                name: pat_name,
                                dbg: _,
                            } => rhs
                                .substitute_basis_alias(pat_name.to_string(), expanded_arg.clone())
                                .expand(env),

                            BasisMacroPattern::BasisLiteral {
                                vec_names: pat_vec_names,
                                dbg: _,
                            } => match arg_prog {
                                // Unfortunately, we don't know if this matches yet.
                                Progress::Partial => Ok((
                                    MetaBasisGenerator::BasisGeneratorMacro {
                                        name: name.to_string(),
                                        arg: Box::new(expanded_arg),
                                        dbg: dbg.clone(),
                                    },
                                    Progress::Partial,
                                )),

                                Progress::Full => match expanded_arg {
                                    MetaBasis::EmptyBasisLiteral { .. }
                                        if pat_vec_names.is_empty() =>
                                    {
                                        rhs.expand(env)
                                    }

                                    MetaBasis::BasisLiteral { vecs: arg_vecs, .. }
                                        if pat_vec_names.len() == arg_vecs.len() =>
                                    {
                                        let mut subst_rhs = rhs.clone();
                                        for (pat_vec_name, arg_vec) in
                                            pat_vec_names.iter().zip(arg_vecs.into_iter())
                                        {
                                            subst_rhs = subst_rhs.substitute_vector_alias(
                                                pat_vec_name.to_string(),
                                                arg_vec,
                                            );
                                        }
                                        subst_rhs.expand(env)
                                    }

                                    // Operand doesn't match or wasn't actually fully expanded
                                    MetaBasis::EmptyBasisLiteral { dbg, .. }
                                    | MetaBasis::BasisLiteral { dbg, .. }
                                    | MetaBasis::BasisAlias { dbg, .. }
                                    | MetaBasis::BasisAliasRec { dbg, .. }
                                    | MetaBasis::BasisBroadcastTensor { dbg, .. }
                                    | MetaBasis::BasisBiTensor { dbg, .. }
                                    | MetaBasis::ApplyBasisGenerator { dbg, .. } => {
                                        Err(LowerError {
                                            kind: LowerErrorKind::Malformed,
                                            dbg: dbg.clone(),
                                        })
                                    }
                                },
                            },
                        })
                } else {
                    // Wrong type of macro or macro def missing
                    Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                }
            }

            MetaBasisGenerator::Revolve { v1, v2, dbg } => {
                v1.expand(env).and_then(|(expanded_v1, v1_prog)| {
                    v2.expand(env).map(|(expanded_v2, v2_prog)| {
                        (
                            MetaBasisGenerator::Revolve {
                                v1: expanded_v1,
                                v2: expanded_v2,
                                dbg: dbg.clone(),
                            },
                            v1_prog.join(v2_prog),
                        )
                    })
                })
            }
        }
    }
}

impl MetaBasis {
    pub fn expand(&self, env: &MacroEnv) -> Result<(MetaBasis, Progress), LowerError> {
        match self {
            MetaBasis::BasisAlias { name, dbg } => {
                match env.aliases.get(name) {
                    Some(AliasBinding::BasisAlias { rhs }) => rhs.expand(env),

                    // Wrong type of alias or alias missing
                    Some(AliasBinding::BasisAliasRec { .. }) | None => Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    }),
                }
            }

            MetaBasis::BasisAliasRec { name, param, dbg } => {
                match env.aliases.get(name) {
                    Some(AliasBinding::BasisAliasRec {
                        base_cases,
                        recursive_step,
                    }) => param.expand(env).and_then(|(expanded_param, param_prog)| {
                        match (&expanded_param, param_prog) {
                            (
                                DimExpr::DimConst {
                                    val,
                                    dbg: dim_const_dbg,
                                },
                                Progress::Full,
                            ) => {
                                if let Some(base_case_basis) = base_cases.get(&val) {
                                    base_case_basis.expand(env)
                                } else if let Some((dim_var_name, rec_basis)) = recursive_step {
                                    let param = DimVar::MacroParam { var_name: dim_var_name.to_string() };
                                    let const_val = DimExpr::DimConst {
                                        val: val.clone(),
                                        dbg: dim_const_dbg.clone(),
                                    };
                                    rec_basis
                                        .substitute_dim_var(param, const_val)
                                        .expand(env)
                                } else {
                                    // Missing recursive step
                                    Err(LowerError {
                                        kind: LowerErrorKind::Malformed,
                                        dbg: dbg.clone(),
                                    })
                                }
                            }
                            _ => Ok((
                                MetaBasis::BasisAliasRec {
                                    name: name.to_string(),
                                    param: expanded_param,
                                    dbg: dbg.clone(),
                                },
                                param_prog,
                            )),
                        }
                    }),

                    // Wrong type of alias or alias missing
                    Some(AliasBinding::BasisAlias { .. }) | None => Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    }),
                }
            }

            MetaBasis::BasisBroadcastTensor { val, factor, dbg } => {
                val.expand(env).and_then(|(expanded_val, val_prog)| {
                    factor.expand(env).and_then(|(expanded_factor, factor_prog)| {
                        match (&expanded_factor, factor_prog) {
                            (DimExpr::DimConst { .. }, Progress::Full) => expanded_factor
                                .extract()
                                .map(|factor_int| {
                                    if factor_int == 0 {
                                        (MetaBasis::EmptyBasisLiteral { dbg: dbg.clone() },
                                         Progress::Full)
                                    } else {
                                        let n_fold_tensor_product = std::iter::repeat(expanded_val)
                                            .take(factor_int)
                                            .reduce(|acc, cloned_val| MetaBasis::BasisBiTensor {
                                                left: Box::new(acc),
                                                right: Box::new(cloned_val),
                                                dbg: dbg.clone()
                                            })
                                            .expect("factor_int > 0, so tensor product should not be empty");
                                        (n_fold_tensor_product, val_prog.join(factor_prog))
                                    }
                                }),

                            _ => Ok((
                                MetaBasis::BasisBroadcastTensor {
                                    val: Box::new(expanded_val),
                                    factor: expanded_factor,
                                    dbg: dbg.clone()
                                },
                                val_prog.join(factor_prog)
                            )),
                        }
                    })
                })
            }

            MetaBasis::BasisLiteral { vecs, dbg } => {
                vecs.iter()
                    .map(|vec| vec.expand(env))
                    .collect::<Result<Vec<_>, _>>().map(|expanded_pairs| {
                        let (expanded_vecs, vec_progs) : (Vec<_>, Vec<_>) = expanded_pairs
                            .into_iter()
                            .unzip();
                        let prog = vec_progs
                            .iter()
                            .fold(Progress::identity(),
                                  |acc, vec_prog| acc.join(*vec_prog));
                        (MetaBasis::BasisLiteral { vecs: expanded_vecs, dbg: dbg.clone() }, prog)
                    })
            }

            MetaBasis::BasisBiTensor { left, right, dbg } => left
                .expand(env)
                .and_then(|(expanded_left, left_prog)|
                    right.expand(env).map(|(expanded_right, right_prog)|
                        (MetaBasis::BasisBiTensor {
                            left: Box::new(expanded_left),
                            right: Box::new(expanded_right),
                            dbg: dbg.clone()
                        },
                        left_prog.join(right_prog))
            )),

            MetaBasis::ApplyBasisGenerator { basis, generator, dbg } => basis
                .expand(env)
                .and_then(|(expanded_basis, basis_prog)|
                    generator.expand(env).map(|(expanded_generator, generator_prog)|
                        (MetaBasis::ApplyBasisGenerator {
                            basis: Box::new(expanded_basis),
                            generator: expanded_generator,
                            dbg: dbg.clone()
                        },
                        basis_prog.join(generator_prog)))),

            MetaBasis::EmptyBasisLiteral { .. } => Ok((self.clone(), Progress::Full)),
        }
    }

    pub fn substitute_basis_alias(&self, basis_alias: String, new_basis: MetaBasis) -> MetaBasis {
        match self {
            MetaBasis::BasisAlias { name, .. } => {
                if *name == basis_alias {
                    new_basis
                } else {
                    self.clone()
                }
            }

            MetaBasis::BasisBroadcastTensor { val, factor, dbg } => {
                MetaBasis::BasisBroadcastTensor {
                    val: Box::new(val.substitute_basis_alias(basis_alias, new_basis)),
                    factor: factor.clone(),
                    dbg: dbg.clone(),
                }
            }

            MetaBasis::BasisBiTensor { left, right, dbg } => MetaBasis::BasisBiTensor {
                left: Box::new(
                    left.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                right: Box::new(right.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaBasis::ApplyBasisGenerator {
                basis,
                generator,
                dbg,
            } => MetaBasis::ApplyBasisGenerator {
                basis: Box::new(
                    basis.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                generator: generator.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            MetaBasis::BasisAliasRec { .. }
            | MetaBasis::BasisLiteral { .. }
            | MetaBasis::EmptyBasisLiteral { .. } => self.clone(),
        }
    }

    pub fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: MetaVector,
    ) -> MetaBasis {
        match self {
            MetaBasis::BasisBroadcastTensor { val, factor, dbg } => {
                MetaBasis::BasisBroadcastTensor {
                    val: Box::new(val.substitute_vector_alias(vector_alias, new_vector)),
                    factor: factor.clone(),
                    dbg: dbg.clone(),
                }
            }

            MetaBasis::BasisLiteral { vecs, dbg } => MetaBasis::BasisLiteral {
                vecs: vecs
                    .iter()
                    .map(|vec| {
                        vec.substitute_vector_alias(vector_alias.to_string(), new_vector.clone())
                    })
                    .collect(),
                dbg: dbg.clone(),
            },

            MetaBasis::BasisBiTensor { left, right, dbg } => MetaBasis::BasisBiTensor {
                left: Box::new(
                    left.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                right: Box::new(right.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaBasis::ApplyBasisGenerator {
                basis,
                generator,
                dbg,
            } => MetaBasis::ApplyBasisGenerator {
                basis: Box::new(
                    basis.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                generator: generator.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            MetaBasis::BasisAlias { .. }
            | MetaBasis::BasisAliasRec { .. }
            | MetaBasis::EmptyBasisLiteral { .. } => self.clone(),
        }
    }

    pub fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> MetaBasis {
        match self {
            MetaBasis::BasisAliasRec { name, param, dbg } => MetaBasis::BasisAliasRec {
                name: name.to_string(),
                param: param.substitute_dim_var(dim_var, new_dim_expr.clone()),
                dbg: dbg.clone(),
            },

            MetaBasis::BasisBroadcastTensor { val, factor, dbg } => {
                MetaBasis::BasisBroadcastTensor {
                    val: val.clone(),
                    factor: factor.substitute_dim_var(dim_var, new_dim_expr.clone()),
                    dbg: dbg.clone(),
                }
            }

            MetaBasis::BasisLiteral { vecs, dbg } => MetaBasis::BasisLiteral {
                vecs: vecs
                    .iter()
                    .map(|vec| vec.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()))
                    .collect(),
                dbg: dbg.clone(),
            },

            MetaBasis::BasisBiTensor { left, right, dbg } => MetaBasis::BasisBiTensor {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaBasis::ApplyBasisGenerator {
                basis,
                generator,
                dbg,
            } => MetaBasis::ApplyBasisGenerator {
                basis: Box::new(basis.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                generator: generator.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaBasis::BasisAlias { .. } | MetaBasis::EmptyBasisLiteral { .. } => self.clone(),
        }
    }
}

impl MetaExpr {
    pub fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> MetaExpr {
        match self {
            MetaExpr::ExprMacro { name, arg, dbg } => MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisMacro { name, arg, dbg } => MetaExpr::BasisMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::BroadcastTensor { val, factor, dbg } => MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                factor: factor.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::Instantiate { name, param, dbg } => MetaExpr::Instantiate {
                name: name.to_string(),
                param: param.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => MetaExpr::Repeat {
                for_each: Box::new(
                    for_each.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                ),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_dim_var(dim_var, new_dim_expr)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            MetaExpr::Adjoint { func, dbg } => MetaExpr::Adjoint {
                func: Box::new(func.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::Pipe { lhs, rhs, dbg } => MetaExpr::Pipe {
                lhs: Box::new(lhs.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                rhs: Box::new(rhs.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::Measure { basis, dbg } => MetaExpr::Measure {
                basis: basis.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::BiTensor { left, right, dbg } => MetaExpr::BiTensor {
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisTranslation { bin, bout, dbg } => MetaExpr::BasisTranslation {
                bin: bin.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                bout: bout.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                ),
                pred: pred.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::NonUniformSuperpos { pairs, dbg } => MetaExpr::NonUniformSuperpos {
                pairs: pairs
                    .iter()
                    .map(|(prob, vec)| {
                        (
                            prob.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                            vec.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                        )
                    })
                    .collect(),
                dbg: dbg.clone(),
            },

            MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                ),
                cond: Box::new(cond.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::QLit { vec } => MetaExpr::QLit {
                vec: vec.substitute_dim_var(dim_var, new_dim_expr),
            },

            MetaExpr::BitLiteral { val, n_bits, dbg } => MetaExpr::BitLiteral {
                val: val.clone(),
                n_bits: n_bits.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            MetaExpr::Variable { .. } | MetaExpr::UnitLiteral { .. } | MetaExpr::Discard { .. } => {
                self.clone()
            }
        }
    }

    pub fn substitute_variable(&self, var_name: String, new_expr: MetaExpr) -> MetaExpr {
        match self {
            MetaExpr::Variable { name, .. } => {
                if *name == var_name {
                    new_expr
                } else {
                    self.clone()
                }
            }

            MetaExpr::ExprMacro { name, arg, dbg } => MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::BroadcastTensor { val, factor, dbg } => MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_variable(var_name, new_expr)),
                factor: factor.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => MetaExpr::Repeat {
                for_each: Box::new(for_each.substitute_variable(var_name, new_expr)),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_variable(var_name, new_expr)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            MetaExpr::Adjoint { func, dbg } => MetaExpr::Adjoint {
                func: Box::new(func.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::Pipe { lhs, rhs, dbg } => MetaExpr::Pipe {
                lhs: Box::new(lhs.substitute_variable(var_name.to_string(), new_expr.clone())),
                rhs: Box::new(rhs.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::BiTensor { left, right, dbg } => MetaExpr::BiTensor {
                left: Box::new(left.substitute_variable(var_name.to_string(), new_expr.clone())),
                right: Box::new(right.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                pred: pred.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                cond: Box::new(cond.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisMacro { .. }
            | MetaExpr::Instantiate { .. }
            | MetaExpr::UnitLiteral { .. }
            | MetaExpr::Measure { .. }
            | MetaExpr::Discard { .. }
            | MetaExpr::BasisTranslation { .. }
            | MetaExpr::NonUniformSuperpos { .. }
            | MetaExpr::QLit { .. }
            | MetaExpr::BitLiteral { .. } => self.clone(),
        }
    }

    pub fn substitute_basis_alias(&self, basis_alias: String, new_basis: MetaBasis) -> MetaExpr {
        match self {
            MetaExpr::ExprMacro { name, arg, dbg } => MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisMacro { name, arg, dbg } => MetaExpr::BasisMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaExpr::BroadcastTensor { val, factor, dbg } => MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_basis_alias(basis_alias, new_basis)),
                factor: factor.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => MetaExpr::Repeat {
                for_each: Box::new(for_each.substitute_basis_alias(basis_alias, new_basis)),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_basis_alias(basis_alias, new_basis)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            MetaExpr::Adjoint { func, dbg } => MetaExpr::Adjoint {
                func: Box::new(func.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaExpr::Pipe { lhs, rhs, dbg } => MetaExpr::Pipe {
                lhs: Box::new(
                    lhs.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                rhs: Box::new(rhs.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaExpr::Measure { basis, dbg } => MetaExpr::Measure {
                basis: basis.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            MetaExpr::BiTensor { left, right, dbg } => MetaExpr::BiTensor {
                left: Box::new(
                    left.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                right: Box::new(right.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisTranslation { bin, bout, dbg } => MetaExpr::BasisTranslation {
                bin: bin.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                bout: bout.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                pred: pred.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                cond: Box::new(cond.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            MetaExpr::Instantiate { .. }
            | MetaExpr::Variable { .. }
            | MetaExpr::UnitLiteral { .. }
            | MetaExpr::Discard { .. }
            | MetaExpr::NonUniformSuperpos { .. }
            | MetaExpr::QLit { .. }
            | MetaExpr::BitLiteral { .. } => self.clone(),
        }
    }

    pub fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: MetaVector,
    ) -> MetaExpr {
        match self {
            MetaExpr::ExprMacro { name, arg, dbg } => MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisMacro { name, arg, dbg } => MetaExpr::BasisMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaExpr::BroadcastTensor { val, factor, dbg } => MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_vector_alias(vector_alias, new_vector)),
                factor: factor.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => MetaExpr::Repeat {
                for_each: Box::new(for_each.substitute_vector_alias(vector_alias, new_vector)),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.clone(),
                dbg: dbg.clone(),
            },

            MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_vector_alias(vector_alias, new_vector)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            MetaExpr::Adjoint { func, dbg } => MetaExpr::Adjoint {
                func: Box::new(func.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaExpr::Pipe { lhs, rhs, dbg } => MetaExpr::Pipe {
                lhs: Box::new(
                    lhs.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                rhs: Box::new(rhs.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaExpr::Measure { basis, dbg } => MetaExpr::Measure {
                basis: basis.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            MetaExpr::BiTensor { left, right, dbg } => MetaExpr::BiTensor {
                left: Box::new(
                    left.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                right: Box::new(right.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaExpr::BasisTranslation { bin, bout, dbg } => MetaExpr::BasisTranslation {
                bin: bin.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                bout: bout.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                pred: pred.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            MetaExpr::NonUniformSuperpos { pairs, dbg } => MetaExpr::NonUniformSuperpos {
                pairs: pairs
                    .iter()
                    .map(|(prob, vec)| {
                        (
                            prob.clone(),
                            vec.substitute_vector_alias(
                                vector_alias.to_string(),
                                new_vector.clone(),
                            ),
                        )
                    })
                    .collect(),
                dbg: dbg.clone(),
            },

            MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                cond: Box::new(cond.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            MetaExpr::QLit { vec } => MetaExpr::QLit {
                vec: vec.substitute_vector_alias(vector_alias, new_vector),
            },

            MetaExpr::Instantiate { .. }
            | MetaExpr::Variable { .. }
            | MetaExpr::UnitLiteral { .. }
            | MetaExpr::Discard { .. }
            | MetaExpr::BitLiteral { .. } => self.clone(),
        }
    }

    pub fn expand(&self, env: &MacroEnv) -> Result<(MetaExpr, Progress), LowerError> {
        match self {
            MetaExpr::ExprMacro { name, arg, dbg } => {
                match env.macros.get(name) {
                    Some(MacroBinding::ExprMacro {
                        lhs_pat:
                            ExprMacroPattern::AnyExpr {
                                name: pat_name,
                                dbg: _,
                            },
                        rhs,
                    }) => {
                        // The progress of expanding the arg doesn't even
                        // matter since the result of rhs.substitute_variable(...)
                        // .expand() will incorporate that progress.
                        arg.expand(env).and_then(|(expanded_arg, _arg_progress)| {
                            rhs.substitute_variable(pat_name.to_string(), expanded_arg.clone())
                                .expand(env)
                        })
                    }

                    // Not defined or param doesn't match
                    None
                    | Some(MacroBinding::BasisMacro { .. })
                    | Some(MacroBinding::BasisGeneratorMacro { .. }) => Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    }),
                }
            }

            MetaExpr::BasisMacro { name, arg, dbg } => {
                if let MetaBasis::BasisAlias {
                    name: arg_alias_name,
                    dbg: arg_dbg,
                } = &**arg
                    && !env.aliases.contains_key(arg_alias_name)
                {
                    MetaExpr::ExprMacro {
                        name: name.to_string(),
                        arg: Box::new(MetaExpr::Variable {
                            name: arg_alias_name.to_string(),
                            dbg: arg_dbg.clone(),
                        }),
                        dbg: dbg.clone(),
                    }
                    .expand(env)
                } else {
                    match env.macros.get(name) {
                        Some(MacroBinding::BasisMacro {
                            lhs_pat:
                                BasisMacroPattern::AnyBasis {
                                    name: pat_name,
                                    dbg: _,
                                },
                            rhs,
                        }) => {
                            // As mentioned above, we can ignore arg_progress
                            arg.expand(env).and_then(|(expanded_arg, _arg_progress)| {
                                rhs.substitute_basis_alias(
                                    pat_name.to_string(),
                                    expanded_arg.clone(),
                                )
                                .expand(env)
                            })
                        }

                        Some(MacroBinding::BasisMacro {
                            lhs_pat:
                                BasisMacroPattern::BasisLiteral {
                                    vec_names: pat_vec_names,
                                    dbg: _,
                                },
                            rhs,
                        }) => arg.expand(env).and_then(|(expanded_arg, arg_progress)| {
                            match arg_progress {
                                // This is unfortunate, but we cannot actually
                                // match yet. Consider fourier[N] where N=1, for
                                // example.
                                Progress::Partial => Ok((
                                    MetaExpr::BasisMacro {
                                        name: name.to_string(),
                                        arg: Box::new(expanded_arg),
                                        dbg: dbg.clone(),
                                    },
                                    Progress::Partial,
                                )),

                                Progress::Full => match expanded_arg {
                                    MetaBasis::EmptyBasisLiteral { .. }
                                        if pat_vec_names.is_empty() =>
                                    {
                                        rhs.expand(env)
                                    }

                                    MetaBasis::BasisLiteral { vecs: arg_vecs, .. }
                                        if arg_vecs.len() == pat_vec_names.len() =>
                                    {
                                        let mut subst_rhs = rhs.clone();
                                        for (pat_vec_name, arg_vec) in
                                            pat_vec_names.iter().zip(arg_vecs.into_iter())
                                        {
                                            subst_rhs = subst_rhs.substitute_vector_alias(
                                                pat_vec_name.to_string(),
                                                arg_vec,
                                            );
                                        }
                                        subst_rhs.expand(env)
                                    }

                                    // Operand doesn't match or wasn't actually fully expanded
                                    MetaBasis::EmptyBasisLiteral { dbg, .. }
                                    | MetaBasis::BasisLiteral { dbg, .. }
                                    | MetaBasis::BasisAlias { dbg, .. }
                                    | MetaBasis::BasisAliasRec { dbg, .. }
                                    | MetaBasis::BasisBroadcastTensor { dbg, .. }
                                    | MetaBasis::BasisBiTensor { dbg, .. }
                                    | MetaBasis::ApplyBasisGenerator { dbg, .. } => {
                                        Err(LowerError {
                                            kind: LowerErrorKind::Malformed,
                                            dbg: dbg.clone(),
                                        })
                                    }
                                },
                            }
                        }),

                        // Not defined or param doesn't match
                        None
                        | Some(MacroBinding::ExprMacro { .. })
                        | Some(MacroBinding::BasisGeneratorMacro { .. }) => Err(LowerError {
                            kind: LowerErrorKind::Malformed,
                            dbg: dbg.clone(),
                        }),
                    }
                }
            }

            MetaExpr::BroadcastTensor { val, factor, dbg } => val
                .expand(env)
                .and_then(|(expanded_val, val_prog)|
                    factor.expand(env).and_then(|(expanded_factor, factor_prog)|
                        match (&expanded_factor, factor_prog) {
                            (DimExpr::DimConst { .. }, Progress::Full) =>
                                expanded_factor.extract().map(|factor_int|
                                    if factor_int == 0 {
                                        (MetaExpr::UnitLiteral { dbg: dbg.clone() }, Progress::Full)
                                    } else {
                                        let n_fold_tensor_product = std::iter::repeat(expanded_val)
                                            .take(factor_int)
                                            .reduce(|acc, cloned_val| MetaExpr::BiTensor {
                                                left: Box::new(acc),
                                                right: Box::new(cloned_val),
                                                dbg: dbg.clone()
                                            })
                                            .expect("factor_int > 0, so tensor product should not be empty");
                                        (n_fold_tensor_product, val_prog.join(factor_prog))
                                    }
                                ),

                            _ => Ok((
                                MetaExpr::BroadcastTensor {
                                    val: Box::new(expanded_val),
                                    factor: expanded_factor,
                                    dbg: dbg.clone()
                                },
                                val_prog.join(factor_prog)
                            )),
                        })),

            MetaExpr::Instantiate { name, param, dbg } => param
                .expand(env)
                .map(|(expanded_param, param_prog)|
                    (MetaExpr::Instantiate {
                        name: name.to_string(),
                        param: expanded_param,
                        dbg: dbg.clone()
                    }, param_prog)),

            MetaExpr::Repeat { for_each, iter_var, upper_bound, dbg } => upper_bound
                .expand(env).and_then(|(expanded_ub, ub_prog)|
                    if let (DimExpr::DimConst { .. }, Progress::Full) = (&expanded_ub, ub_prog) {
                        expanded_ub.extract().and_then(|ub_int|
                            if ub_int == 0 {
                                // TODO: if we ever have an identity node,
                                //       insert it here instead of erroring
                                Err(LowerError {
                                    kind: LowerErrorKind::Malformed,
                                    dbg: dbg.clone(),
                                })
                            } else {
                                let var = DimVar::MacroParam { var_name: iter_var.to_string() };
                                let first = for_each
                                    .substitute_dim_var(
                                        var.clone(),
                                        DimExpr::DimConst {
                                            val: 0.into(),
                                            dbg: dbg.clone()
                                        });

                                std::iter::repeat(for_each)
                                    .take(ub_int-1)
                                    .enumerate()
                                    .fold(first, |acc, (i, cloned_for_each)| {
                                        let rhs = cloned_for_each
                                            .substitute_dim_var(
                                                var.clone(),
                                                DimExpr::DimConst {
                                                    val: (i+1).into(),
                                                    dbg: dbg.clone()
                                                });

                                        MetaExpr::Pipe {
                                            lhs: Box::new(acc),
                                            rhs: Box::new(rhs),
                                            dbg: dbg.clone(),
                                        }
                                    })
                                    .expand(env)
                            })
                    } else {
                        Ok((MetaExpr::Repeat {
                            for_each: for_each.clone(),
                            iter_var: iter_var.to_string(),
                            upper_bound: expanded_ub,
                            dbg: dbg.clone(),
                        }, ub_prog))
                    }),

            MetaExpr::EmbedClassical { func, embed_kind, dbg } => func
                .expand(env)
                .map(|(expanded_func, func_prog)|
                    (MetaExpr::EmbedClassical {
                        func: Box::new(expanded_func),
                        embed_kind: *embed_kind,
                        dbg: dbg.clone()
                    },
                    func_prog)
                ),

            MetaExpr::Adjoint { func, dbg } => func
                .expand(env)
                .map(|(expanded_func, func_prog)|
                    (MetaExpr::Adjoint {
                        func: Box::new(expanded_func),
                        dbg: dbg.clone()
                    },
                    func_prog)
                ),

            MetaExpr::Pipe { lhs, rhs, dbg } => lhs
                .expand(env)
                .and_then(|(expanded_lhs, lhs_prog)|
                    rhs.expand(env)
                        .map(|(expanded_rhs, rhs_prog)|
                            (MetaExpr::Pipe {
                                lhs: Box::new(expanded_lhs),
                                rhs: Box::new(expanded_rhs),
                                dbg: dbg.clone(),
                            }, lhs_prog.join(rhs_prog))
                        )),

            MetaExpr::Measure { basis, dbg } => basis
                .expand(env)
                .map(|(expanded_basis, basis_prog)|
                    (MetaExpr::Measure {
                        basis: expanded_basis,
                        dbg: dbg.clone()
                    },
                    basis_prog)
                ),

            MetaExpr::BiTensor { left, right, dbg } => left
                .expand(env)
                .and_then(|(expanded_left, left_prog)|
                    right.expand(env)
                        .map(|(expanded_right, right_prog)|
                            (MetaExpr::BiTensor {
                                left: Box::new(expanded_left),
                                right: Box::new(expanded_right),
                                dbg: dbg.clone(),
                            }, left_prog.join(right_prog))
                        )),

            MetaExpr::BasisTranslation { bin, bout, dbg } => bin
                .expand(env)
                .and_then(|(expanded_bin, bin_prog)|
                    bout.expand(env)
                        .map(|(expanded_bout, bout_prog)|
                            (MetaExpr::BasisTranslation {
                                bin: expanded_bin,
                                bout: expanded_bout,
                                dbg: dbg.clone(),
                            }, bin_prog.join(bout_prog))
                        )),

            MetaExpr::Predicated { then_func, else_func, pred, dbg } => then_func
                .expand(env)
                .and_then(|(expanded_then, then_prog)|
                    else_func.expand(env)
                        .and_then(|(expanded_else, else_prog)|
                            pred.expand(env)
                                .map(|(expanded_pred, pred_prog)|
                                    (
                                        MetaExpr::Predicated {
                                            then_func: Box::new(expanded_then),
                                            else_func: Box::new(expanded_else),
                                            pred: expanded_pred,
                                            dbg: dbg.clone(),
                                        },
                                        then_prog.join(else_prog)
                                            .join(pred_prog)
                                    )
                                )
                        )
                ),

            MetaExpr::NonUniformSuperpos { pairs, dbg } => pairs
                .iter()
                .map(|(prob, vec)|
                    prob.expand(env)
                        .and_then(|(expanded_prob, prob_prog)|
                            vec.expand(env)
                                .map(|(expanded_vec, vec_prog)|
                                    ((expanded_prob, expanded_vec), prob_prog.join(vec_prog)))))
                .collect::<Result<Vec<_>, _>>()
                .map(|pair_pairs| {
                    let (expanded_pairs, progs): (Vec<_>, Vec<_>) = pair_pairs.into_iter().unzip();
                    let prog = progs
                        .into_iter()
                        .fold(Progress::identity(), |acc, prog| acc.join(prog));
                    let expanded_superpos = MetaExpr::NonUniformSuperpos {
                        pairs: expanded_pairs,
                        dbg: dbg.clone(),
                    };
                    (expanded_superpos, prog)
                }),

            MetaExpr::Conditional { then_expr, else_expr, cond, dbg } => then_expr
                .expand(env)
                .and_then(|(expanded_then, then_prog)|
                    else_expr.expand(env)
                        .and_then(|(expanded_else, else_prog)|
                            cond.expand(env)
                                .map(|(expanded_cond, cond_prog)|
                                    (
                                        MetaExpr::Conditional {
                                            then_expr: Box::new(expanded_then),
                                            else_expr: Box::new(expanded_else),
                                            cond: Box::new(expanded_cond),
                                            dbg: dbg.clone(),
                                        },
                                        then_prog.join(else_prog)
                                            .join(cond_prog)
                                    )
                                )
                        )
                ),

            MetaExpr::QLit { vec } => vec
                .expand(env)
                .map(|(expanded_vec, vec_prog)|
                    (
                        MetaExpr::QLit { vec: expanded_vec },
                        vec_prog,
                    )
                ),

            MetaExpr::BitLiteral { val, n_bits, dbg } => n_bits
                .expand(env)
                .map(|(expanded_n_bits, n_bits_prog)|
                    (
                        MetaExpr::BitLiteral {
                            val: val.clone(),
                            n_bits: expanded_n_bits,
                            dbg: dbg.clone()
                        },
                        n_bits_prog,
                    )
                ),

            MetaExpr::Variable { .. }
            | MetaExpr::UnitLiteral { .. }
            | MetaExpr::Discard { .. } => Ok((self.clone(), Progress::Full)),
        }
    }
}

impl Expandable for MetaStmt {
    fn expand(&self, env: &mut MacroEnv) -> Result<(MetaStmt, Progress), LowerError> {
        match self {
            MetaStmt::ExprMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => {
                if env
                    .macros
                    .insert(
                        lhs_name.to_string(),
                        MacroBinding::ExprMacro {
                            lhs_pat: lhs_pat.clone(),
                            rhs: rhs.clone(),
                        },
                    )
                    .is_some()
                {
                    // Duplicate macro
                    Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), Progress::Full))
                }
            }

            MetaStmt::BasisMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => {
                if env
                    .macros
                    .insert(
                        lhs_name.to_string(),
                        MacroBinding::BasisMacro {
                            lhs_pat: lhs_pat.clone(),
                            rhs: rhs.clone(),
                        },
                    )
                    .is_some()
                {
                    // Duplicate macro
                    Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), Progress::Full))
                }
            }

            MetaStmt::BasisGeneratorMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => {
                if env
                    .macros
                    .insert(
                        lhs_name.to_string(),
                        MacroBinding::BasisGeneratorMacro {
                            lhs_pat: lhs_pat.clone(),
                            rhs: rhs.clone(),
                        },
                    )
                    .is_some()
                {
                    // Duplicate macro
                    Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), Progress::Full))
                }
            }

            MetaStmt::VectorSymbolDef { lhs, rhs, dbg } => {
                if env.vec_symbols.insert(*lhs, rhs.clone()).is_some() {
                    // Duplicate vector symbol def
                    Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), Progress::Full))
                }
            }

            MetaStmt::BasisAliasDef { lhs, rhs, dbg } => {
                if env
                    .aliases
                    .insert(
                        lhs.to_string(),
                        AliasBinding::BasisAlias { rhs: rhs.clone() },
                    )
                    .is_some()
                {
                    // Duplicate basis alias
                    Err(LowerError {
                        kind: LowerErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), Progress::Full))
                }
            }

            MetaStmt::BasisAliasRecDef {
                lhs,
                param,
                rhs,
                dbg,
            } => {
                if let Some(existing_alias) = env.aliases.get_mut(lhs) {
                    match existing_alias {
                        AliasBinding::BasisAlias { .. } => {
                            // Duplicate basis alias
                            Err(LowerError {
                                kind: LowerErrorKind::Malformed,
                                dbg: dbg.clone(),
                            })
                        }

                        AliasBinding::BasisAliasRec {
                            base_cases,
                            recursive_step,
                        } => {
                            match param {
                                RecDefParam::Base(constant) => {
                                    if base_cases.insert(constant.clone(), rhs.clone()).is_none() {
                                        Ok((self.clone(), Progress::Full))
                                    } else {
                                        // Duplicate basis alias
                                        Err(LowerError {
                                            kind: LowerErrorKind::Malformed,
                                            dbg: dbg.clone(),
                                        })
                                    }
                                }
                                RecDefParam::Rec(dim_var_name) => {
                                    if recursive_step.is_none() {
                                        *recursive_step =
                                            Some((dim_var_name.to_string(), rhs.clone()));
                                        Ok((self.clone(), Progress::Full))
                                    } else {
                                        // Only 1 recursive def is allowed
                                        Err(LowerError {
                                            kind: LowerErrorKind::Malformed,
                                            dbg: dbg.clone(),
                                        })
                                    }
                                }
                            }
                        }
                    }
                } else {
                    let base_cases = {
                        let mut base_cases = HashMap::new();
                        if let RecDefParam::Base(constant) = param {
                            base_cases.insert(constant.clone(), rhs.clone());
                        }
                        base_cases
                    };
                    let recursive_step = if let RecDefParam::Rec(dim_var_name) = param {
                        Some((dim_var_name.to_string(), rhs.clone()))
                    } else {
                        None
                    };
                    let binding = AliasBinding::BasisAliasRec {
                        base_cases,
                        recursive_step,
                    };
                    let inserted = env.aliases.insert(lhs.to_string(), binding).is_none();
                    assert!(inserted, "alias didn't exist but now it does!");
                    Ok((self.clone(), Progress::Full))
                }
            }

            MetaStmt::Expr { expr } => expr.expand(env).map(|(expanded_expr, progress)| {
                (
                    MetaStmt::Expr {
                        expr: expanded_expr,
                    },
                    progress,
                )
            }),

            MetaStmt::Assign { lhs, rhs, dbg } => {
                rhs.expand(env).map(|(expanded_expr, progress)| {
                    (
                        MetaStmt::Assign {
                            lhs: lhs.to_string(),
                            rhs: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }

            MetaStmt::UnpackAssign { lhs, rhs, dbg } => {
                rhs.expand(env).map(|(expanded_expr, progress)| {
                    (
                        MetaStmt::UnpackAssign {
                            lhs: lhs.clone(),
                            rhs: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }

            MetaStmt::Return { val, dbg } => val.expand(env).map(|(expanded_expr, progress)| {
                (
                    MetaStmt::Return {
                        val: expanded_expr,
                        dbg: dbg.clone(),
                    },
                    progress,
                )
            }),
        }
    }
}
