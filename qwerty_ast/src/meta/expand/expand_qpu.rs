use crate::{
    ast::{angle_is_approx_zero, usize_try_into_angle},
    error::{LowerError, LowerErrorKind},
    meta::{
        DimExpr, DimVar, Progress,
        expand::{AliasBinding, Expandable, MacroBinding, MacroEnv},
        qpu::{
            BasisMacroPattern, ExprMacroPattern, FloatExpr, MetaBasis, MetaBasisGenerator,
            MetaExpr, MetaStmt, MetaVector, RecDefParam, float_expr, meta_basis,
            meta_basis_generator, meta_expr, meta_stmt, meta_vector,
        },
    },
};
use qwerty_ast_macros::rebuild;
use std::collections::HashMap;

impl FloatExpr {
    /// Only added due to limitations in the `gen_rebuild` framework (since
    /// this is used in [`MetaExpr::NonUniformSuperpos`] and
    /// [`MetaExpr::Ensemble`]), and `substitute_vector_alias()` is defined on
    /// `MetaExpr` with `recurse_attrs`.
    pub fn substitute_vector_alias(self, _vector_alias: &str, _new_vector: &MetaVector) -> Self {
        self
    }

    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> FloatExpr {
        rebuild!(FloatExpr, self, substitute_dim_var, dim_var, new_dim_expr)
    }

    pub(crate) fn expand_rewriter(
        self,
        _env: &MacroEnv,
        children_progress: Progress,
    ) -> Result<(Self, Progress), LowerError> {
        match (self, children_progress) {
            (FloatExpr::FloatDimExpr { expr, dbg }, Progress::Full) => {
                let expr_int = expr.extract()?;
                usize_try_into_angle(expr_int)
                    .ok_or_else(|| LowerError {
                        kind: LowerErrorKind::IntegerTooBig {
                            offender: expr_int.into(),
                        },
                        dbg: dbg.clone(),
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
            }

            (FloatExpr::FloatSum { left, right, dbg }, Progress::Full) => {
                let left_val = left.extract()?;
                let right_val = right.extract()?;
                let val = left_val + right_val;
                Ok((FloatExpr::FloatConst { val, dbg }, Progress::Full))
            }

            (FloatExpr::FloatProd { left, right, dbg }, Progress::Full) => {
                let left_val = left.extract()?;
                let right_val = right.extract()?;
                let val = left_val * right_val;
                Ok((FloatExpr::FloatConst { val, dbg }, Progress::Full))
            }

            (FloatExpr::FloatDiv { left, right, dbg }, Progress::Full) => {
                let left_val = left.extract()?;
                let right_val = right.extract()?;
                if angle_is_approx_zero(right_val) {
                    Err(LowerError {
                        kind: LowerErrorKind::DivisionByZero,
                        dbg,
                    })
                } else {
                    let val = left_val / right_val;
                    Ok((FloatExpr::FloatConst { val, dbg }, Progress::Full))
                }
            }

            (FloatExpr::FloatNeg { val, dbg }, Progress::Full) => {
                let val = val.extract()?;
                Ok((FloatExpr::FloatConst { val, dbg }, Progress::Full))
            }

            (unfinished, Progress::Partial) => Ok((unfinished, Progress::Partial)),

            (done @ FloatExpr::FloatConst { .. }, _) => Ok((done, Progress::Full)),
        }
    }

    pub fn expand(self, env: &MacroEnv) -> Result<(FloatExpr, Progress), LowerError> {
        rebuild!(FloatExpr, self, expand, env)
    }
}

impl MetaVector {
    pub(crate) fn substitute_vector_alias_rewriter(
        self,
        vector_alias: &str,
        new_vector: &MetaVector,
    ) -> Self {
        match self {
            MetaVector::VectorAlias { name, dbg } => {
                if name == *vector_alias {
                    new_vector.clone()
                } else {
                    MetaVector::VectorAlias { name, dbg }
                }
            }

            other => other,
        }
    }

    pub fn substitute_vector_alias(
        self,
        vector_alias: &str,
        new_vector: &MetaVector,
    ) -> MetaVector {
        rebuild!(
            MetaVector,
            self,
            substitute_vector_alias,
            vector_alias,
            new_vector
        )
    }

    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(MetaVector, self, substitute_dim_var, dim_var, new_dim_expr)
    }

    pub fn expand(self, env: &MacroEnv) -> Result<(MetaVector, Progress), LowerError> {
        rebuild!(MetaVector, self, expand, env)
    }

    pub(crate) fn expand_rewriter(
        self,
        env: &MacroEnv,
        children_progress: Progress,
    ) -> Result<(Self, Progress), LowerError> {
        match (self, children_progress) {
            // Only substitution can remove this
            (alias @ MetaVector::VectorAlias { .. }, _) => Ok((alias, Progress::Partial)),

            (MetaVector::VectorSymbol { sym, dbg }, _) => {
                if let Some(vec) = env.vec_symbols.get(&sym) {
                    vec.clone().expand(env)
                } else {
                    Err(LowerError {
                        kind: LowerErrorKind::UndefinedQubitSymbol { sym },
                        dbg: dbg,
                    })
                }
            }

            // TODO: It's only important that factor_int is fully expanded. Is
            //       this condition too strict?
            (MetaVector::VectorBroadcastTensor { val, factor, dbg }, Progress::Full) => {
                let factor_int = factor.extract()?;
                if factor_int == 0 {
                    Ok((MetaVector::VectorUnit { dbg }, Progress::Full))
                } else {
                    let n_fold_tensor_product = std::iter::repeat(*val)
                        .take(factor_int)
                        .reduce(|acc, cloned_val| MetaVector::VectorBiTensor {
                            left: Box::new(acc),
                            right: Box::new(cloned_val),
                            dbg: dbg.clone(),
                        })
                        .expect("factor_int > 0, so tensor product should not be empty");
                    Ok((n_fold_tensor_product, Progress::Full))
                }
            }

            (unfinished @ MetaVector::VectorBroadcastTensor { .. }, Progress::Partial) => {
                Ok((unfinished, Progress::Partial))
            }

            (
                done @ (MetaVector::ZeroVector { .. }
                | MetaVector::OneVector { .. }
                | MetaVector::PadVector { .. }
                | MetaVector::TargetVector { .. }
                | MetaVector::VectorUnit { .. }),
                _,
            ) => Ok((done, Progress::Full)),

            (
                other @ (MetaVector::VectorTilt { .. }
                | MetaVector::UniformVectorSuperpos { .. }
                | MetaVector::VectorBiTensor { .. }),
                _,
            ) => Ok((other, children_progress)),
        }
    }
}

impl MetaBasisGenerator {
    pub fn substitute_basis_alias(
        self,
        basis_alias: &str,
        new_basis: &MetaBasis,
    ) -> MetaBasisGenerator {
        rebuild!(
            MetaBasisGenerator,
            self,
            substitute_basis_alias,
            basis_alias,
            new_basis
        )
    }

    pub fn substitute_vector_alias(
        self,
        vector_alias: &str,
        new_vector: &MetaVector,
    ) -> MetaBasisGenerator {
        rebuild!(
            MetaBasisGenerator,
            self,
            substitute_vector_alias,
            vector_alias,
            new_vector
        )
    }

    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(
            MetaBasisGenerator,
            self,
            substitute_dim_var,
            dim_var,
            new_dim_expr
        )
    }

    pub fn expand(self, env: &MacroEnv) -> Result<(MetaBasisGenerator, Progress), LowerError> {
        rebuild!(MetaBasisGenerator, self, expand, env)
    }

    pub(crate) fn expand_rewriter(
        self,
        env: &MacroEnv,
        children_progress: Progress,
    ) -> Result<(MetaBasisGenerator, Progress), LowerError> {
        match self {
            MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                match env.macros.get(&name) {
                    Some(MacroBinding::BasisGeneratorMacro { lhs_pat, rhs }) => {
                        match lhs_pat {
                            BasisMacroPattern::AnyBasis {
                                name: pat_name,
                                dbg: _,
                            } => rhs
                                .clone()
                                .substitute_basis_alias(pat_name, &*arg)
                                .expand(env),

                            BasisMacroPattern::BasisLiteral {
                                vec_names: pat_vec_names,
                                dbg: _,
                            } => match children_progress {
                                // Unfortunately, we don't know if this matches yet.
                                Progress::Partial => Ok((
                                    MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg },
                                    Progress::Partial,
                                )),

                                Progress::Full => match *arg {
                                    MetaBasis::EmptyBasisLiteral { .. }
                                        if pat_vec_names.is_empty() =>
                                    {
                                        rhs.clone().expand(env)
                                    }

                                    MetaBasis::BasisLiteral { vecs: arg_vecs, .. }
                                        if pat_vec_names.len() == arg_vecs.len() =>
                                    {
                                        pat_vec_names
                                            .iter()
                                            .zip(arg_vecs.iter())
                                            .fold(
                                                rhs.clone(),
                                                |subst_rhs, (pat_vec_name, arg_vec)| {
                                                    subst_rhs.substitute_vector_alias(
                                                        pat_vec_name,
                                                        arg_vec,
                                                    )
                                                },
                                            )
                                            .expand(env)
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
                                            kind: LowerErrorKind::MacroDoesNotMatch {
                                                macro_name: name,
                                            },
                                            dbg,
                                        })
                                    }
                                },
                            },
                        }
                    }

                    Some(_) => Err(LowerError {
                        kind: LowerErrorKind::WrongMacroKind { macro_name: name },
                        dbg,
                    }),

                    None => Err(LowerError {
                        kind: LowerErrorKind::UndefinedMacro { macro_name: name },
                        dbg,
                    }),
                }
            }

            other @ MetaBasisGenerator::Revolve { .. } => Ok((other, children_progress)),
        }
    }
}

impl MetaBasis {
    pub fn expand(self, env: &MacroEnv) -> Result<(MetaBasis, Progress), LowerError> {
        rebuild!(MetaBasis, self, expand, env)
    }

    pub(crate) fn expand_rewriter(
        self,
        env: &MacroEnv,
        children_progress: Progress,
    ) -> Result<(MetaBasis, Progress), LowerError> {
        match (self, children_progress) {
            (MetaBasis::BasisAlias { name, dbg }, _) => match env.aliases.get(&name) {
                Some(AliasBinding::BasisAlias { rhs }) => rhs.clone().expand(env),

                Some(AliasBinding::BasisAliasRec { .. }) => Err(LowerError {
                    kind: LowerErrorKind::WrongAliasKind { alias_name: name },
                    dbg,
                }),

                None => Err(LowerError {
                    kind: LowerErrorKind::UndefinedAlias { alias_name: name },
                    dbg,
                }),
            },

            (MetaBasis::BasisAliasRec { name, param, dbg }, Progress::Full) => {
                match env.aliases.get(&name) {
                    Some(AliasBinding::BasisAliasRec {
                        base_cases,
                        recursive_step,
                    }) => {
                        if let DimExpr::DimConst {
                            val,
                            dbg: dim_const_dbg,
                        } = param
                        {
                            if let Some(base_case_basis) = base_cases.get(&val) {
                                base_case_basis.clone().expand(env)
                            } else if let Some((dim_var_name, rec_basis)) = recursive_step {
                                let param = DimVar::MacroParam {
                                    var_name: dim_var_name.to_string(),
                                };
                                let const_val = DimExpr::DimConst {
                                    val: val.clone(),
                                    dbg: dim_const_dbg.clone(),
                                };
                                rec_basis
                                    .clone()
                                    .substitute_dim_var(&param, &const_val)
                                    .expand(env)
                            } else {
                                // Missing recursive step
                                Err(LowerError {
                                    kind: LowerErrorKind::MissingAliasRecursiveStep {
                                        alias_name: name,
                                    },
                                    dbg,
                                })
                            }
                        } else {
                            // Fully expanded DimExpr should be a constant
                            Err(LowerError {
                                kind: LowerErrorKind::Malformed,
                                dbg,
                            })
                        }
                    }

                    Some(AliasBinding::BasisAlias { .. }) => Err(LowerError {
                        kind: LowerErrorKind::WrongAliasKind { alias_name: name },
                        dbg,
                    }),

                    None => Err(LowerError {
                        kind: LowerErrorKind::UndefinedAlias { alias_name: name },
                        dbg,
                    }),
                }
            }

            (MetaBasis::BasisBroadcastTensor { val, factor, dbg }, Progress::Full) => {
                let factor_int = factor.extract()?;
                if factor_int == 0 {
                    Ok((MetaBasis::EmptyBasisLiteral { dbg }, Progress::Full))
                } else {
                    let n_fold_tensor_product = std::iter::repeat(*val)
                        .take(factor_int)
                        .reduce(|acc, cloned_val| MetaBasis::BasisBiTensor {
                            left: Box::new(acc),
                            right: Box::new(cloned_val),
                            dbg: dbg.clone(),
                        })
                        .expect("factor_int > 0, so tensor product should not be empty");
                    Ok((n_fold_tensor_product, Progress::Full))
                }
            }

            (
                unfinished @ (MetaBasis::BasisAliasRec { .. }
                | MetaBasis::BasisBroadcastTensor { .. }),
                Progress::Partial,
            ) => Ok((unfinished, Progress::Partial)),

            (done @ MetaBasis::EmptyBasisLiteral { .. }, _) => Ok((done, Progress::Full)),

            (
                other @ (MetaBasis::BasisLiteral { .. }
                | MetaBasis::BasisBiTensor { .. }
                | MetaBasis::ApplyBasisGenerator { .. }),
                _,
            ) => Ok((other, children_progress)),
        }
    }

    pub fn substitute_basis_alias(self, basis_alias: &str, new_basis: &MetaBasis) -> MetaBasis {
        rebuild!(
            MetaBasis,
            self,
            substitute_basis_alias,
            basis_alias,
            new_basis
        )
    }

    pub(crate) fn substitute_basis_alias_rewriter(
        self,
        basis_alias: &str,
        new_basis: &MetaBasis,
    ) -> MetaBasis {
        match self {
            MetaBasis::BasisAlias { name, dbg } => {
                if name == *basis_alias {
                    new_basis.clone()
                } else {
                    MetaBasis::BasisAlias { name, dbg }
                }
            }

            other => other,
        }
    }

    pub fn substitute_vector_alias(self, vector_alias: &str, new_vector: &MetaVector) -> MetaBasis {
        rebuild!(
            MetaBasis,
            self,
            substitute_vector_alias,
            vector_alias,
            new_vector
        )
    }

    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(MetaBasis, self, substitute_dim_var, dim_var, new_dim_expr)
    }
}

impl MetaExpr {
    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(MetaExpr, self, substitute_dim_var, dim_var, new_dim_expr)
    }

    pub fn substitute_variable(self, var_name: &str, new_expr: &MetaExpr) -> MetaExpr {
        rebuild!(MetaExpr, self, substitute_variable, var_name, new_expr)
    }

    pub(crate) fn substitute_variable_rewriter(
        self,
        var_name: &str,
        new_expr: &MetaExpr,
    ) -> MetaExpr {
        match self {
            MetaExpr::Variable { name, dbg } => {
                if name == *var_name {
                    new_expr.clone()
                } else {
                    MetaExpr::Variable { name, dbg }
                }
            }

            other => other,
        }
    }

    pub fn substitute_basis_alias(self, basis_alias: &str, new_basis: &MetaBasis) -> MetaExpr {
        rebuild!(
            MetaExpr,
            self,
            substitute_basis_alias,
            basis_alias,
            new_basis
        )
    }

    pub fn substitute_vector_alias(self, vector_alias: &str, new_vector: &MetaVector) -> MetaExpr {
        rebuild!(
            MetaExpr,
            self,
            substitute_vector_alias,
            vector_alias,
            new_vector
        )
    }

    /// Called by the `gen_rebuild` attribute macro invoked in `meta/qpu.rs`.
    pub(crate) fn expand_rewriter(
        self,
        env: &MacroEnv,
        children_progress: Progress,
    ) -> Result<(Self, Progress), LowerError> {
        match self {
            MetaExpr::ExprMacro { name, arg, dbg } => {
                match env.macros.get(&name) {
                    Some(MacroBinding::ExprMacro {
                        lhs_pat:
                            ExprMacroPattern::AnyExpr {
                                name: pat_name,
                                dbg: _,
                            },
                        rhs,
                    }) => {
                        // The progress of expanding the arg (incorporated in
                        // children_progress) doesn't even matter since the
                        // result of rhs.substitute_variable(...).expand() will
                        // incorporate that progress.
                        rhs.clone().substitute_variable(pat_name, &*arg).expand(env)
                    }

                    Some(MacroBinding::BasisMacro { .. })
                    | Some(MacroBinding::BasisGeneratorMacro { .. }) => Err(LowerError {
                        kind: LowerErrorKind::WrongMacroKind { macro_name: name },
                        dbg,
                    }),

                    None => Err(LowerError {
                        kind: LowerErrorKind::UndefinedMacro { macro_name: name },
                        dbg,
                    }),
                }
            }

            MetaExpr::BasisMacro { name, arg, dbg } => {
                if let MetaBasis::BasisAlias {
                    name: arg_alias_name,
                    dbg: arg_dbg,
                } = &arg
                    && !env.aliases.contains_key(arg_alias_name)
                {
                    MetaExpr::ExprMacro {
                        name: name,
                        arg: Box::new(MetaExpr::Variable {
                            name: arg_alias_name.to_string(),
                            dbg: arg_dbg.clone(),
                        }),
                        dbg: dbg,
                    }
                    .expand(env)
                } else {
                    match env.macros.get(&name) {
                        Some(MacroBinding::BasisMacro {
                            lhs_pat:
                                BasisMacroPattern::AnyBasis {
                                    name: pat_name,
                                    dbg: _,
                                },
                            rhs,
                        }) => {
                            // As mentioned above, we can ignore the progress
                            // of the arg (children)
                            rhs.clone()
                                .substitute_basis_alias(pat_name, &arg)
                                .expand(env)
                        }

                        Some(MacroBinding::BasisMacro {
                            lhs_pat:
                                BasisMacroPattern::BasisLiteral {
                                    vec_names: pat_vec_names,
                                    dbg: _,
                                },
                            rhs,
                        }) => {
                            match children_progress {
                                // This is unfortunate, but we cannot actually
                                // match yet. Consider fourier[N] when
                                // eventually N=1, for example.
                                Progress::Partial => {
                                    Ok((MetaExpr::BasisMacro { name, arg, dbg }, Progress::Partial))
                                }

                                Progress::Full => match arg {
                                    MetaBasis::EmptyBasisLiteral { .. }
                                        if pat_vec_names.is_empty() =>
                                    {
                                        rhs.clone().expand(env)
                                    }

                                    MetaBasis::BasisLiteral { vecs: arg_vecs, .. }
                                        if arg_vecs.len() == pat_vec_names.len() =>
                                    {
                                        pat_vec_names
                                            .iter()
                                            .zip(arg_vecs.iter())
                                            .fold(
                                                rhs.clone(),
                                                |subst_rhs, (pat_vec_name, arg_vec)| {
                                                    subst_rhs.substitute_vector_alias(
                                                        pat_vec_name,
                                                        arg_vec,
                                                    )
                                                },
                                            )
                                            .expand(env)
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
                                            kind: LowerErrorKind::MacroDoesNotMatch {
                                                macro_name: name,
                                            },
                                            dbg,
                                        })
                                    }
                                },
                            }
                        }

                        Some(MacroBinding::ExprMacro { .. })
                        | Some(MacroBinding::BasisGeneratorMacro { .. }) => Err(LowerError {
                            kind: LowerErrorKind::WrongMacroKind { macro_name: name },
                            dbg,
                        }),

                        None => Err(LowerError {
                            kind: LowerErrorKind::UndefinedMacro { macro_name: name },
                            dbg,
                        }),
                    }
                }
            }

            MetaExpr::BroadcastTensor { val, factor, dbg } => match (factor, children_progress) {
                (const_factor @ DimExpr::DimConst { .. }, Progress::Full) => {
                    let factor_int = const_factor.extract()?;
                    if factor_int == 0 {
                        Ok((MetaExpr::UnitLiteral { dbg }, Progress::Full))
                    } else {
                        let n_fold_tensor_product = std::iter::repeat(*val)
                            .take(factor_int)
                            .reduce(|acc, cloned_val| MetaExpr::BiTensor {
                                left: Box::new(acc),
                                right: Box::new(cloned_val),
                                dbg: dbg.clone(),
                            })
                            .expect("factor_int > 0, so tensor product should not be empty");
                        Ok((n_fold_tensor_product, Progress::Full))
                    }
                }

                (nonconst_factor, _) => Ok((
                    MetaExpr::BroadcastTensor {
                        val,
                        factor: nonconst_factor,
                        dbg,
                    },
                    children_progress,
                )),
            },

            // TODO: create a lambda node and then expand to
            //       (lambda q: for_each | for_each | ... | for_each)
            repeat @ MetaExpr::Repeat { .. } => Ok((repeat, Progress::Partial)),

            // TODO: This is a hack. Remove this and do the TODO above
            MetaExpr::Pipe { lhs, rhs, dbg } => match *rhs {
                MetaExpr::Repeat {
                    for_each,
                    iter_var,
                    // TODO: Use progress like normal. Or better yet,
                    //       obliterate this dumb hack entirely (see todo above)
                    upper_bound: upper_bound @ DimExpr::DimConst { .. },
                    dbg,
                } => {
                    let ub_int = upper_bound.extract()?;
                    let var = DimVar::MacroParam { var_name: iter_var };

                    std::iter::repeat(for_each)
                        .take(ub_int)
                        .enumerate()
                        .fold(*lhs, |acc, (i, cloned_for_each)| {
                            let rhs = cloned_for_each.substitute_dim_var(
                                &var,
                                &DimExpr::DimConst {
                                    val: i.into(),
                                    dbg: dbg.clone(),
                                },
                            );

                            MetaExpr::Pipe {
                                lhs: Box::new(acc),
                                rhs: Box::new(rhs),
                                dbg: dbg.clone(),
                            }
                        })
                        .expand(env)
                }

                other => Ok((
                    MetaExpr::Pipe {
                        lhs,
                        rhs: Box::new(other),
                        dbg,
                    },
                    children_progress,
                )),
            },

            other @ (MetaExpr::Variable { .. }
            | MetaExpr::UnitLiteral { .. }
            | MetaExpr::Discard { .. }
            | MetaExpr::Instantiate { .. }
            | MetaExpr::EmbedClassical { .. }
            | MetaExpr::Adjoint { .. }
            | MetaExpr::Measure { .. }
            | MetaExpr::BiTensor { .. }
            | MetaExpr::Tilt { .. }
            | MetaExpr::BasisTranslation { .. }
            | MetaExpr::Predicated { .. }
            | MetaExpr::NonUniformSuperpos { .. }
            | MetaExpr::Ensemble { .. }
            | MetaExpr::Conditional { .. }
            | MetaExpr::QLit { .. }
            | MetaExpr::BitLiteral { .. }) => Ok((other, children_progress)),
        }
    }

    pub fn expand(self, env: &MacroEnv) -> Result<(Self, Progress), LowerError> {
        rebuild!(MetaExpr, self, expand, env)
    }
}

impl MetaStmt {
    pub(crate) fn expand_rewriter(
        self,
        env: &mut MacroEnv,
        children_progress: Progress,
    ) -> Result<(Self, Progress), LowerError> {
        match &self {
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
                    Err(LowerError {
                        kind: LowerErrorKind::DuplicateMacroDef {
                            macro_name: lhs_name.to_string(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok(())
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
                    Err(LowerError {
                        kind: LowerErrorKind::DuplicateMacroDef {
                            macro_name: lhs_name.to_string(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok(())
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
                    Err(LowerError {
                        kind: LowerErrorKind::DuplicateMacroDef {
                            macro_name: lhs_name.to_string(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok(())
                }
            }

            MetaStmt::VectorSymbolDef { lhs, rhs, dbg } => {
                if env.vec_symbols.insert(*lhs, rhs.clone()).is_some() {
                    // Duplicate vector symbol def
                    Err(LowerError {
                        kind: LowerErrorKind::DuplicateQubitSymbolDef { sym: *lhs },
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok(())
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
                    Err(LowerError {
                        kind: LowerErrorKind::DuplicateBasisAliasDef {
                            alias_name: lhs.to_string(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok(())
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
                        AliasBinding::BasisAlias { .. } => Err(LowerError {
                            kind: LowerErrorKind::DuplicateBasisAliasDef {
                                alias_name: lhs.to_string(),
                            },
                            dbg: dbg.clone(),
                        }),

                        AliasBinding::BasisAliasRec {
                            base_cases,
                            recursive_step,
                        } => {
                            match &param {
                                RecDefParam::Base(constant) => {
                                    if base_cases.insert(constant.clone(), rhs.clone()).is_none() {
                                        Ok(())
                                    } else {
                                        Err(LowerError {
                                            kind: LowerErrorKind::DuplicateBasisAliasDef {
                                                alias_name: lhs.to_string(),
                                            },
                                            dbg: dbg.clone(),
                                        })
                                    }
                                }
                                RecDefParam::Rec(dim_var_name) => {
                                    if recursive_step.is_none() {
                                        *recursive_step =
                                            Some((dim_var_name.to_string(), rhs.clone()));
                                        Ok(())
                                    } else {
                                        // Only 1 recursive def is allowed
                                        Err(LowerError {
                                            kind: LowerErrorKind::DuplicateBasisAliasRecDef {
                                                alias_name: lhs.to_string(),
                                            },
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
                        if let RecDefParam::Base(constant) = &param {
                            base_cases.insert(constant.clone(), rhs.clone());
                        }
                        base_cases
                    };
                    let recursive_step = if let RecDefParam::Rec(dim_var_name) = &param {
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
                    Ok(())
                }
            }

            MetaStmt::Expr { .. }
            | MetaStmt::Assign { .. }
            | MetaStmt::UnpackAssign { .. }
            | MetaStmt::Return { .. } => Ok(()),
        }?;

        Ok((self, children_progress))
    }
}

impl Expandable for MetaStmt {
    fn expand(self, env: &mut MacroEnv) -> Result<(MetaStmt, Progress), LowerError> {
        rebuild!(MetaStmt, self, expand, env)
    }
}
