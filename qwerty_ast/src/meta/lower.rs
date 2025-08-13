use crate::{
    ast,
    error::{ExtractError, ExtractErrorKind},
    meta::MetaProgram,
};

/// Allows expansion/inference to report on whether it is completed yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// More rounds of expansion/inference are needed.
    Partial,

    /// Expansion/inference is finished. Extraction will work successfully.
    Full,
}

impl Progress {
    /// Returns the progress `P` such that `P.join(P') == P'` for any `P'`.
    pub fn identity() -> Self {
        Progress::Full
    }

    /// If a parent node contains two sub-nodes who reported progress
    /// `self` and `other`, then returns the progress of the parent node.
    pub fn join(self, other: Progress) -> Progress {
        match (self, other) {
            (Progress::Full, Progress::Full) => Progress::Full,
            (Progress::Partial, _) | (_, Progress::Partial) => Progress::Partial,
        }
    }

    /// Returns true if expansion/inference is finished.
    pub fn is_finished(self) -> bool {
        self == Progress::Full
    }
}

impl MetaProgram {
    pub fn lower(&self) -> Result<ast::Program, ExtractError> {
        let mut init_program = self.clone();
        loop {
            let (program, expansion_progress) = init_program.expand()?;
            let (program, infer_progress) = program.infer_and_instantiate()?;

            if expansion_progress.is_finished() && infer_progress.is_finished() {
                return program.extract();
            } else if init_program == program {
                return Err(ExtractError {
                    kind: ExtractErrorKind::Stuck,
                    dbg: self.dbg.clone(),
                });
            } else {
                init_program = program;
                continue;
            }
        }
    }
}
