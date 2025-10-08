use crate::compile::{
    CompileConfig, CompileError, QirProfile, Target, compile_meta_ast, translate_to_llvm_ir,
};
use dashu::integer::UBig;
use qwerty_ast::meta::MetaProgram;

mod backend;

pub enum Backend {
    QirRunner,
    Qiree,
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
        Backend::Qiree => {
            let cfg = CompileConfig {
                target: Target::Qir(QirProfile::Base),
                dump: debug,
            };
            let mlir_module = compile_meta_ast(prog, func_name, &cfg)?;
            let llvm_module = translate_to_llvm_ir(mlir_module, debug)?;
            Ok(backend::qiree::run_llvm_module(
                llvm_module,
                func_name,
                num_shots,
            ))
        }
    }
}
