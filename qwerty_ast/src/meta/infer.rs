use crate::{
    error::ExtractError,
    meta::{MetaProgram, Progress},
};

impl MetaProgram {
    pub fn infer_and_instantiate(&self) -> Result<(MetaProgram, Progress), ExtractError> {
        // TODO: actually do inference here
        Ok((self.clone(), Progress::Full))
    }
}
