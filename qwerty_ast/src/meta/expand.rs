use crate::{
    error::ExtractError,
    meta::{
        qpu::{self, MetaBasis, MetaBasisGenerator, MetaExpr, MetaVector},
        DimExpr, MetaFunc, MetaFunctionDef, MetaProgram, MetaType,
    },
};
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
    ExprMacro(MetaExpr),
    BasisGeneratorMacro(MetaBasisGenerator),
}

struct MetaEnv {
    aliases: HashMap<String, AliasBinding>,
    macros: HashMap<String, MacroBinding>,
    dim_vars: HashMap<String, DimExpr>,
    vec_symbols: HashMap<char, MetaVector>,
}

impl MetaEnv {
    fn new() -> MetaEnv {
        MetaEnv {
            aliases: HashMap::new(),
            macros: HashMap::new(),
            dim_vars: HashMap::new(),
            vec_symbols: HashMap::new(),
        }
    }
}

impl MetaFunctionDef<qpu::MetaStmt> {
    fn expand(&self) -> Result<(MetaFunctionDef<qpu::MetaStmt>, ExpansionProgress), ExtractError> {
        todo!("MetaFunctionDef<qpu::MetaStmt>::expand()")
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
