use crate::{
    error::{ExtractError, ExtractErrorKind},
    meta::{
        DimExpr, MetaFunc, MetaFunctionDef, MetaProgram,
        qpu::{
            self, BasisMacroPattern, ExprMacroPattern, MetaBasis, MetaBasisGenerator, MetaExpr,
            MetaVector,
        },
    },
};
use dashu::integer::IBig;
use std::collections::HashMap;

/// Allows expansion to report on whether it is completed yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionProgress {
    /// More rounds of expansion are needed.
    Partial,

    /// Expansion is finished. Extraction will work successfully.
    Full,
}

impl ExpansionProgress {
    /// Returns the expansion progress `P` such that `P.join(P') == P'` for any `P'`.
    fn identity() -> Self {
        ExpansionProgress::Full
    }

    /// If a parent node contains two sub-nodes who reported expansion progress
    /// `self` and `other`, then returns the expansion progress of the parent node.
    fn join(self, other: ExpansionProgress) -> ExpansionProgress {
        match (self, other) {
            (ExpansionProgress::Full, ExpansionProgress::Full) => ExpansionProgress::Full,
            (ExpansionProgress::Partial, _) | (_, ExpansionProgress::Partial) => {
                ExpansionProgress::Partial
            }
        }
    }
}

enum AliasBinding {
    BasisAlias {
        rhs: MetaBasis,
    },
    BasisAliasRec {
        base_cases: HashMap<IBig, MetaBasis>,
        recursive_step: Option<(String, MetaBasis)>,
    },
}

enum MacroBinding {
    ExprMacro {
        lhs_pat: ExprMacroPattern,
        rhs: MetaExpr,
    },
    BasisMacro {
        lhs_pat: BasisMacroPattern,
        rhs: MetaExpr,
    },
    BasisGeneratorMacro {
        lhs_pat: BasisMacroPattern,
        rhs: MetaBasisGenerator,
    },
}

enum DimVarValue {
    Unknown,
    Known(IBig),
}

struct MacroEnv {
    aliases: HashMap<String, AliasBinding>,
    macros: HashMap<String, MacroBinding>,
    dim_vars: HashMap<String, DimVarValue>,
    vec_symbols: HashMap<char, MetaVector>,
}

impl MacroEnv {
    fn new() -> MacroEnv {
        MacroEnv {
            aliases: HashMap::new(),
            macros: HashMap::new(),
            dim_vars: HashMap::new(),
            vec_symbols: HashMap::new(),
        }
    }
}

impl DimExpr {
    fn expand(&self, env: &MacroEnv) -> Result<(DimExpr, ExpansionProgress), ExtractError> {
        match self {
            DimExpr::DimVar { name, dbg } => {
                if let Some(DimVarValue::Known(val)) = env.dim_vars.get(name) {
                    Ok((
                        DimExpr::DimConst {
                            val: val.clone(),
                            dbg: dbg.clone(),
                        },
                        ExpansionProgress::Full,
                    ))
                } else {
                    Ok((self.clone(), ExpansionProgress::Partial))
                }
            }

            DimExpr::DimSum { left, right, dbg } => {
                left.expand(env).and_then(|(expanded_left, left_prog)| {
                    right.expand(env).map(|(expanded_right, right_prog)| {
                        match (&expanded_left, &expanded_right, left_prog.join(right_prog)) {
                            (
                                DimExpr::DimConst { val: left_val, .. },
                                DimExpr::DimConst { val: right_val, .. },
                                prog @ ExpansionProgress::Full,
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
                                prog @ ExpansionProgress::Full,
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

            DimExpr::DimNeg { val, dbg } => {
                val.expand(env)
                    .map(|(expanded_val, val_prog)| match (&expanded_val, val_prog) {
                        (DimExpr::DimConst { val: val_val, .. }, ExpansionProgress::Full) => (
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

            DimExpr::DimConst { .. } => Ok((self.clone(), ExpansionProgress::Full)),
        }
    }
}

impl qpu::MetaVector {
    fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: qpu::MetaVector,
    ) -> qpu::MetaVector {
        match self {
            qpu::MetaVector::VectorAlias { name, .. } => {
                if *name == vector_alias {
                    new_vector
                } else {
                    self.clone()
                }
            }

            qpu::MetaVector::VectorBroadcastTensor { val, factor, dbg } => {
                qpu::MetaVector::VectorBroadcastTensor {
                    val: Box::new(val.substitute_vector_alias(vector_alias, new_vector)),
                    factor: factor.clone(),
                    dbg: dbg.clone(),
                }
            }

            qpu::MetaVector::VectorTilt { q, angle_deg, dbg } => qpu::MetaVector::VectorTilt {
                q: Box::new(q.substitute_vector_alias(vector_alias, new_vector)),
                angle_deg: angle_deg.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaVector::UniformVectorSuperpos { q1, q2, dbg } => {
                qpu::MetaVector::UniformVectorSuperpos {
                    q1: Box::new(
                        q1.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                    ),
                    q2: Box::new(q2.substitute_vector_alias(vector_alias, new_vector)),
                    dbg: dbg.clone(),
                }
            }

            qpu::MetaVector::VectorBiTensor { left, right, dbg } => {
                qpu::MetaVector::VectorBiTensor {
                    left: Box::new(
                        left.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                    ),
                    right: Box::new(right.substitute_vector_alias(vector_alias, new_vector)),
                    dbg: dbg.clone(),
                }
            }

            qpu::MetaVector::VectorSymbol { .. }
            | qpu::MetaVector::ZeroVector { .. }
            | qpu::MetaVector::OneVector { .. }
            | qpu::MetaVector::PadVector { .. }
            | qpu::MetaVector::TargetVector { .. }
            | qpu::MetaVector::VectorUnit { .. } => self.clone(),
        }
    }
}

impl qpu::MetaBasisGenerator {
    fn substitute_basis_alias(
        &self,
        basis_alias: String,
        new_basis: qpu::MetaBasis,
    ) -> qpu::MetaBasisGenerator {
        match self {
            qpu::MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                qpu::MetaBasisGenerator::BasisGeneratorMacro {
                    name: name.to_string(),
                    arg: Box::new(arg.substitute_basis_alias(basis_alias, new_basis)),
                    dbg: dbg.clone(),
                }
            }

            qpu::MetaBasisGenerator::Revolve { .. } => self.clone(),
        }
    }

    fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: qpu::MetaVector,
    ) -> qpu::MetaBasisGenerator {
        match self {
            qpu::MetaBasisGenerator::BasisGeneratorMacro { name, arg, dbg } => {
                qpu::MetaBasisGenerator::BasisGeneratorMacro {
                    name: name.to_string(),
                    arg: Box::new(arg.substitute_vector_alias(vector_alias, new_vector)),
                    dbg: dbg.clone(),
                }
            }

            qpu::MetaBasisGenerator::Revolve { v1, v2, dbg } => qpu::MetaBasisGenerator::Revolve {
                v1: v1.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                v2: v2.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },
        }
    }
}

impl qpu::MetaBasis {
    fn expand(&self, env: &MacroEnv) -> Result<(qpu::MetaBasis, ExpansionProgress), ExtractError> {
        match self {
            qpu::MetaBasis::BasisAlias { name, dbg } => {
                match env.aliases.get(name) {
                    Some(AliasBinding::BasisAlias { rhs }) => rhs.expand(env),

                    // Wrong type of alias or alias missing
                    Some(AliasBinding::BasisAliasRec { .. }) | None => Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    }),
                }
            }

            qpu::MetaBasis::BasisAliasRec { name, param, dbg } => {
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
                                ExpansionProgress::Full,
                            ) => {
                                if let Some(base_case_basis) = base_cases.get(&val) {
                                    base_case_basis.expand(env)
                                } else if let Some((dim_var_name, rec_basis)) = recursive_step {
                                    rec_basis
                                        .substitute_dim_var(
                                            dim_var_name.to_string(),
                                            DimExpr::DimConst {
                                                val: val.clone(),
                                                dbg: dim_const_dbg.clone(),
                                            },
                                        )
                                        .expand(env)
                                } else {
                                    // Missing recursive step
                                    Err(ExtractError {
                                        kind: ExtractErrorKind::Malformed,
                                        dbg: dbg.clone(),
                                    })
                                }
                            }
                            _ => Ok((
                                qpu::MetaBasis::BasisAliasRec {
                                    name: name.to_string(),
                                    param: expanded_param,
                                    dbg: dbg.clone(),
                                },
                                param_prog,
                            )),
                        }
                    }),

                    // Wrong type of alias or alias missing
                    Some(AliasBinding::BasisAlias { .. }) | None => Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    }),
                }
            }

            _ => todo!("qpu::MetaBasis::expand()"),
        }
    }

    fn substitute_basis_alias(
        &self,
        basis_alias: String,
        new_basis: qpu::MetaBasis,
    ) -> qpu::MetaBasis {
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

    fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: qpu::MetaVector,
    ) -> qpu::MetaBasis {
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

    fn substitute_dim_var(&self, dim_var_name: String, new_dim_expr: DimExpr) -> qpu::MetaBasis {
        todo!("MetaBasis::substitute_dim_var")
    }
}

impl qpu::MetaExpr {
    fn substitute_variable(&self, var_name: String, new_expr: qpu::MetaExpr) -> qpu::MetaExpr {
        match self {
            qpu::MetaExpr::Variable { name, .. } => {
                if *name == var_name {
                    new_expr
                } else {
                    self.clone()
                }
            }

            qpu::MetaExpr::ExprMacro { name, arg, dbg } => qpu::MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BroadcastTensor { val, factor, dbg } => qpu::MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_variable(var_name, new_expr)),
                factor: factor.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => qpu::MetaExpr::Repeat {
                for_each: Box::new(for_each.substitute_variable(var_name, new_expr)),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => qpu::MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_variable(var_name, new_expr)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Adjoint { func, dbg } => qpu::MetaExpr::Adjoint {
                func: Box::new(func.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Pipe { lhs, rhs, dbg } => qpu::MetaExpr::Pipe {
                lhs: Box::new(lhs.substitute_variable(var_name.to_string(), new_expr.clone())),
                rhs: Box::new(rhs.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BiTensor { left, right, dbg } => qpu::MetaExpr::BiTensor {
                left: Box::new(left.substitute_variable(var_name.to_string(), new_expr.clone())),
                right: Box::new(right.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => qpu::MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                pred: pred.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => qpu::MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_variable(var_name.to_string(), new_expr.clone()),
                ),
                cond: Box::new(cond.substitute_variable(var_name, new_expr)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BasisMacro { .. }
            | qpu::MetaExpr::Instantiate { .. }
            | qpu::MetaExpr::UnitLiteral { .. }
            | qpu::MetaExpr::Measure { .. }
            | qpu::MetaExpr::Discard { .. }
            | qpu::MetaExpr::BasisTranslation { .. }
            | qpu::MetaExpr::NonUniformSuperpos { .. }
            | qpu::MetaExpr::QLit { .. }
            | qpu::MetaExpr::BitLiteral { .. } => self.clone(),
        }
    }

    fn substitute_basis_alias(
        &self,
        basis_alias: String,
        new_basis: qpu::MetaBasis,
    ) -> qpu::MetaExpr {
        match self {
            qpu::MetaExpr::ExprMacro { name, arg, dbg } => qpu::MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BasisMacro { name, arg, dbg } => qpu::MetaExpr::BasisMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BroadcastTensor { val, factor, dbg } => qpu::MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_basis_alias(basis_alias, new_basis)),
                factor: factor.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => qpu::MetaExpr::Repeat {
                for_each: Box::new(for_each.substitute_basis_alias(basis_alias, new_basis)),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => qpu::MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_basis_alias(basis_alias, new_basis)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Adjoint { func, dbg } => qpu::MetaExpr::Adjoint {
                func: Box::new(func.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Pipe { lhs, rhs, dbg } => qpu::MetaExpr::Pipe {
                lhs: Box::new(
                    lhs.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                rhs: Box::new(rhs.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Measure { basis, dbg } => qpu::MetaExpr::Measure {
                basis: basis.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BiTensor { left, right, dbg } => qpu::MetaExpr::BiTensor {
                left: Box::new(
                    left.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                right: Box::new(right.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BasisTranslation { bin, bout, dbg } => qpu::MetaExpr::BasisTranslation {
                bin: bin.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                bout: bout.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => qpu::MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                pred: pred.substitute_basis_alias(basis_alias, new_basis),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => qpu::MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_basis_alias(basis_alias.to_string(), new_basis.clone()),
                ),
                cond: Box::new(cond.substitute_basis_alias(basis_alias, new_basis)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Instantiate { .. }
            | qpu::MetaExpr::Variable { .. }
            | qpu::MetaExpr::UnitLiteral { .. }
            | qpu::MetaExpr::Discard { .. }
            | qpu::MetaExpr::NonUniformSuperpos { .. }
            | qpu::MetaExpr::QLit { .. }
            | qpu::MetaExpr::BitLiteral { .. } => self.clone(),
        }
    }

    fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: qpu::MetaVector,
    ) -> qpu::MetaExpr {
        match self {
            qpu::MetaExpr::ExprMacro { name, arg, dbg } => qpu::MetaExpr::ExprMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BasisMacro { name, arg, dbg } => qpu::MetaExpr::BasisMacro {
                name: name.to_string(),
                arg: Box::new(arg.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BroadcastTensor { val, factor, dbg } => qpu::MetaExpr::BroadcastTensor {
                val: Box::new(val.substitute_vector_alias(vector_alias, new_vector)),
                factor: factor.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => qpu::MetaExpr::Repeat {
                for_each: Box::new(for_each.substitute_vector_alias(vector_alias, new_vector)),
                iter_var: iter_var.to_string(),
                upper_bound: upper_bound.clone(),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => qpu::MetaExpr::EmbedClassical {
                func: Box::new(func.substitute_vector_alias(vector_alias, new_vector)),
                embed_kind: *embed_kind,
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Adjoint { func, dbg } => qpu::MetaExpr::Adjoint {
                func: Box::new(func.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Pipe { lhs, rhs, dbg } => qpu::MetaExpr::Pipe {
                lhs: Box::new(
                    lhs.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                rhs: Box::new(rhs.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Measure { basis, dbg } => qpu::MetaExpr::Measure {
                basis: basis.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BiTensor { left, right, dbg } => qpu::MetaExpr::BiTensor {
                left: Box::new(
                    left.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                right: Box::new(right.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::BasisTranslation { bin, bout, dbg } => qpu::MetaExpr::BasisTranslation {
                bin: bin.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                bout: bout.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => qpu::MetaExpr::Predicated {
                then_func: Box::new(
                    then_func.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                else_func: Box::new(
                    else_func.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                pred: pred.substitute_vector_alias(vector_alias, new_vector),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::NonUniformSuperpos { pairs, dbg } => qpu::MetaExpr::NonUniformSuperpos {
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

            qpu::MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => qpu::MetaExpr::Conditional {
                then_expr: Box::new(
                    then_expr.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                else_expr: Box::new(
                    else_expr.substitute_vector_alias(vector_alias.to_string(), new_vector.clone()),
                ),
                cond: Box::new(cond.substitute_vector_alias(vector_alias, new_vector)),
                dbg: dbg.clone(),
            },

            qpu::MetaExpr::QLit { vec } => qpu::MetaExpr::QLit {
                vec: vec.substitute_vector_alias(vector_alias, new_vector),
            },

            qpu::MetaExpr::Instantiate { .. }
            | qpu::MetaExpr::Variable { .. }
            | qpu::MetaExpr::UnitLiteral { .. }
            | qpu::MetaExpr::Discard { .. }
            | qpu::MetaExpr::BitLiteral { .. } => self.clone(),
        }
    }

    fn expand(&self, env: &MacroEnv) -> Result<(qpu::MetaExpr, ExpansionProgress), ExtractError> {
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
                    | Some(MacroBinding::BasisGeneratorMacro { .. }) => Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
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
                                ExpansionProgress::Partial => Ok((
                                    MetaExpr::BasisMacro {
                                        name: name.to_string(),
                                        arg: Box::new(expanded_arg),
                                        dbg: dbg.clone(),
                                    },
                                    ExpansionProgress::Partial,
                                )),

                                ExpansionProgress::Full => match expanded_arg {
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
                                            pat_vec_names.into_iter().zip(arg_vecs.into_iter())
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
                                        Err(ExtractError {
                                            kind: ExtractErrorKind::Malformed,
                                            dbg: dbg.clone(),
                                        })
                                    }
                                },
                            }
                        }),

                        // Not defined or param doesn't match
                        None
                        | Some(MacroBinding::ExprMacro { .. })
                        | Some(MacroBinding::BasisGeneratorMacro { .. }) => Err(ExtractError {
                            kind: ExtractErrorKind::Malformed,
                            dbg: dbg.clone(),
                        }),
                    }
                }
            }

            _ => todo!("MetaExpr::expand()"),
        }
    }
}

impl qpu::MetaStmt {
    fn expand(
        &self,
        env: &mut MacroEnv,
    ) -> Result<(qpu::MetaStmt, ExpansionProgress), ExtractError> {
        match self {
            qpu::MetaStmt::ExprMacroDef {
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
                    Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), ExpansionProgress::Full))
                }
            }

            qpu::MetaStmt::BasisMacroDef {
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
                    Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), ExpansionProgress::Full))
                }
            }

            qpu::MetaStmt::BasisGeneratorMacroDef {
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
                    Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), ExpansionProgress::Full))
                }
            }

            qpu::MetaStmt::VectorSymbolDef { lhs, rhs, dbg } => {
                if env.vec_symbols.insert(*lhs, rhs.clone()).is_some() {
                    // Duplicate vector symbol def
                    Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), ExpansionProgress::Full))
                }
            }

            qpu::MetaStmt::BasisAliasDef { lhs, rhs, dbg } => {
                if env
                    .aliases
                    .insert(
                        lhs.to_string(),
                        AliasBinding::BasisAlias { rhs: rhs.clone() },
                    )
                    .is_some()
                {
                    // Duplicate basis alias
                    Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                } else {
                    Ok((self.clone(), ExpansionProgress::Full))
                }
            }

            qpu::MetaStmt::BasisAliasRecDef {
                lhs,
                param,
                rhs,
                dbg,
            } => {
                if let Some(existing_alias) = env.aliases.get_mut(lhs) {
                    match existing_alias {
                        AliasBinding::BasisAlias { .. } => {
                            // Duplicate basis alias
                            Err(ExtractError {
                                kind: ExtractErrorKind::Malformed,
                                dbg: dbg.clone(),
                            })
                        }

                        AliasBinding::BasisAliasRec {
                            base_cases,
                            recursive_step,
                        } => {
                            match param {
                                qpu::RecDefParam::Base(constant) => {
                                    if base_cases.insert(constant.clone(), rhs.clone()).is_none() {
                                        Ok((self.clone(), ExpansionProgress::Full))
                                    } else {
                                        // Duplicate basis alias
                                        Err(ExtractError {
                                            kind: ExtractErrorKind::Malformed,
                                            dbg: dbg.clone(),
                                        })
                                    }
                                }
                                qpu::RecDefParam::Rec(dim_var_name) => {
                                    if recursive_step.is_none() {
                                        *recursive_step =
                                            Some((dim_var_name.to_string(), rhs.clone()));
                                        Ok((self.clone(), ExpansionProgress::Full))
                                    } else {
                                        // Only 1 recursive def is allowed
                                        Err(ExtractError {
                                            kind: ExtractErrorKind::Malformed,
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
                        if let qpu::RecDefParam::Base(constant) = param {
                            base_cases.insert(constant.clone(), rhs.clone());
                        }
                        base_cases
                    };
                    let recursive_step = if let qpu::RecDefParam::Rec(dim_var_name) = param {
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
                    Ok((self.clone(), ExpansionProgress::Full))
                }
            }

            qpu::MetaStmt::Expr { expr } => expr.expand(env).map(|(expanded_expr, progress)| {
                (
                    qpu::MetaStmt::Expr {
                        expr: expanded_expr,
                    },
                    progress,
                )
            }),

            qpu::MetaStmt::Assign { lhs, rhs, dbg } => {
                rhs.expand(env).map(|(expanded_expr, progress)| {
                    (
                        qpu::MetaStmt::Assign {
                            lhs: lhs.to_string(),
                            rhs: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }

            qpu::MetaStmt::UnpackAssign { lhs, rhs, dbg } => {
                rhs.expand(env).map(|(expanded_expr, progress)| {
                    (
                        qpu::MetaStmt::UnpackAssign {
                            lhs: lhs.clone(),
                            rhs: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }

            qpu::MetaStmt::Return { val, dbg } => {
                val.expand(env).map(|(expanded_expr, progress)| {
                    (
                        qpu::MetaStmt::Return {
                            val: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }
        }
    }
}

impl MetaFunctionDef<qpu::MetaStmt> {
    fn expand(&self) -> Result<(MetaFunctionDef<qpu::MetaStmt>, ExpansionProgress), ExtractError> {
        let MetaFunctionDef {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dim_vars,
            dbg,
        } = self;

        let mut env = MacroEnv::new();
        for dim_var in dim_vars {
            if env
                .dim_vars
                .insert(dim_var.to_string(), DimVarValue::Unknown)
                .is_some()
            {
                // Duplicate dimvar
                return Err(ExtractError {
                    kind: ExtractErrorKind::Malformed,
                    dbg: dbg.clone(),
                });
            }
        }

        let expanded_pairs = body
            .iter()
            .map(|stmt| stmt.expand(&mut env))
            .collect::<Result<Vec<(qpu::MetaStmt, ExpansionProgress)>, ExtractError>>()?;
        let (expanded_stmts, progresses): (Vec<_>, Vec<_>) = expanded_pairs.into_iter().unzip();
        let progress = progresses
            .into_iter()
            .fold(ExpansionProgress::identity(), |acc, stmt| acc.join(stmt));

        let expanded_func_def = MetaFunctionDef {
            name: name.to_string(),
            args: args.clone(),
            ret_type: ret_type.clone(),
            body: expanded_stmts,
            is_rev: *is_rev,
            dim_vars: dim_vars.clone(),
            dbg: dbg.clone(),
        };
        Ok((expanded_func_def, progress))
    }
}

impl MetaFunc {
    fn expand(&self) -> Result<(MetaFunc, ExpansionProgress), ExtractError> {
        match self {
            MetaFunc::Qpu(qpu_func_def) => qpu_func_def
                .expand()
                .map(|(expanded_func_def, prog)| (MetaFunc::Qpu(expanded_func_def), prog)),

            // TODO: actually expand classical functions instead of lying here
            MetaFunc::Classical(classical_func_def) => Ok((
                MetaFunc::Classical(classical_func_def.clone()),
                ExpansionProgress::Full,
            )),
        }
    }
}

impl MetaProgram {
    /// Try to expand as many metaQwerty constructs in this program, returning
    /// a new one.
    pub fn expand(&self) -> Result<(MetaProgram, ExpansionProgress), ExtractError> {
        let MetaProgram { funcs, dbg } = self;
        let funcs_pairs = funcs
            .iter()
            .map(MetaFunc::expand)
            .collect::<Result<Vec<(MetaFunc, ExpansionProgress)>, ExtractError>>()?;
        let (expanded_funcs, progresses): (Vec<_>, Vec<_>) = funcs_pairs.into_iter().unzip();
        let progress = progresses
            .into_iter()
            .fold(ExpansionProgress::identity(), |acc, prog| acc.join(prog));

        Ok((
            MetaProgram {
                funcs: expanded_funcs,
                dbg: dbg.clone(),
            },
            progress,
        ))
    }
}
