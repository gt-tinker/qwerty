use crate::{
    error::{ExtractError, ExtractErrorKind},
    meta::{
        DimExpr, MetaFunc, MetaFunctionDef, MetaProgram, MetaType,
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
    BasisAlias(MetaBasis),
    BasisAliasRec(HashMap<DimExpr, MetaBasis>),
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

impl qpu::MetaBasis {
    fn expand(&self, env: &MacroEnv) -> Result<(qpu::MetaBasis, ExpansionProgress), ExtractError> {
        todo!("MetaBasis::expand()")
    }

    fn substitute_basis_alias(
        &self,
        basis_alias: String,
        new_basis: qpu::MetaBasis,
    ) -> qpu::MetaBasis {
        todo!("MetaBasis::substitute_basis_alias()")
    }

    fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: qpu::MetaVector,
    ) -> qpu::MetaBasis {
        todo!("MetaBasis::substitute_vector_alias()")
    }
}

impl qpu::MetaExpr {
    fn substitute_variable(&self, var_name: String, new_expr: qpu::MetaExpr) -> qpu::MetaExpr {
        todo!("MetaExpr::substitute_variable()")
    }

    fn substitute_basis_alias(
        &self,
        basis_alias: String,
        new_basis: qpu::MetaBasis,
    ) -> qpu::MetaExpr {
        todo!("MetaExpr::substitute_basis_alias()")
    }

    fn substitute_vector_alias(
        &self,
        vector_alias: String,
        new_vector: qpu::MetaVector,
    ) -> qpu::MetaExpr {
        todo!("MetaExpr::substitute_vector_alias()")
    }

    fn expand(&self, env: &MacroEnv) -> Result<(qpu::MetaExpr, ExpansionProgress), ExtractError> {
        match self {
            MetaExpr::ExprMacro { name, arg, dbg } => {
                match env.macros.get(name) {
                    Some(MacroBinding::ExprMacro {
                        lhs_pat:
                            ExprMacroPattern::AnyExpr {
                                name: pat_name,
                                dbg: pat_dbg,
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
                                    dbg: pat_dbg,
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
                                    dbg: pat_dbg,
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
                    .insert(lhs.to_string(), AliasBinding::BasisAlias(rhs.clone()))
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
                        AliasBinding::BasisAlias(_) =>
                        // Duplicate basis alias
                        {
                            Err(ExtractError {
                                kind: ExtractErrorKind::Malformed,
                                dbg: dbg.clone(),
                            })
                        }

                        AliasBinding::BasisAliasRec(defs) => {
                            if defs.insert(param.clone(), rhs.clone()).is_some() {
                                // Duplicate basis alias
                                Err(ExtractError {
                                    kind: ExtractErrorKind::Malformed,
                                    dbg: dbg.clone(),
                                })
                            } else {
                                Ok((qpu::MetaStmt::trivial(dbg.clone()), ExpansionProgress::Full))
                            }
                        }
                    }
                } else {
                    let mut defs = HashMap::new();
                    defs.insert(param.clone(), rhs.clone());
                    let existed = env
                        .aliases
                        .insert(lhs.to_string(), AliasBinding::BasisAliasRec(defs))
                        .is_some();
                    assert!(!existed, "alias didn't exist but now it does!");
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
