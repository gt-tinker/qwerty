use crate::compile::{
    CompileConfig, CompileError, QirProfile, Target, compile_meta_ast, translate_to_llvm_ir,
};
use dashu::integer::UBig;
use qwerty_ast::meta::MetaProgram;

mod backend;

#[derive(Debug, Clone)]
pub enum Backend {
    /// JIT and run locally against qir-runner's QIR backend
    QirRunner,
    /// A particular accelerator on QIR-EE
    Qiree(String),
}

pub struct ShotResult {
    pub bits: UBig,
    pub num_bits: usize,
    pub count: usize,
}

pub fn run_meta_ast(
    prog: &MetaProgram,
    func_name: &str,
    backend: Backend,
    num_shots: usize,
    debug: bool,
) -> Result<Vec<ShotResult>, CompileError> {
    match backend {
        Backend::QirRunner => {
            let cfg = CompileConfig {
                target: Target::Qir(QirProfile::Unrestricted),
                dump: debug,
            };
            let module = compile_meta_ast(prog, func_name, &cfg)?;
            Ok(backend::qir_runner::run_mlir_module(
                module, func_name, num_shots,
            ))
        }
        Backend::Qiree(acc) => {
            let cfg = CompileConfig {
                target: Target::Qir(QirProfile::Base),
                dump: debug,
            };
            let mlir_module = compile_meta_ast(prog, func_name, &cfg)?;
            let llvm_module = translate_to_llvm_ir(mlir_module, debug)?;
            backend::qiree::run_llvm_module(llvm_module, func_name, acc, num_shots).map_err(
                |qiree_err| CompileError::Message(format!("QIR-EE error: {}", qiree_err), None),
            )
        }
    }
}
