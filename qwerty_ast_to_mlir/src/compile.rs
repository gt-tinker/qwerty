use crate::{
    ctx::{LLVM_CTX, MLIR_CTX},
    lower::ast_program_to_mlir,
};
use melior::{
    Error,
    dialect::{qcirc, qwerty},
    ir::{Module, OperationLike, operation::OperationPrintingFlags, symbol_table::SymbolTable},
    pass::{PassIrPrintingOptions, PassManager, transform},
    target::llvm_ir::{LLVMModule, translate_module},
};
use qwerty_ast::{
    dbg::DebugLoc,
    error::{LowerError, TypeError},
    meta::MetaProgram,
};
use std::{env, fs, path::PathBuf};

const QWERTY_DEBUG_DIR: &str = "qwerty-debug";
const MLIR_DUMP_SUBDIR: &str = "mlir";
const INIT_MLIR_FILENAME: &str = "initial.mlir";
const QWERTY_AST_FILENAME: &str = "qwerty_ast.py";
const LLVM_IR_FILENAME: &str = "module.ll";

#[derive(Debug, Clone, Copy)]
pub enum QirProfile {
    Unrestricted,
    Base,
}

#[derive(Debug, Clone, Copy)]
pub enum Target {
    Qir(QirProfile),
    OpenQasm,
}

#[derive(Debug, Clone, Copy)]
pub struct CompileConfig {
    pub target: Target,
    pub dump: bool,
}

fn run_passes(module: &mut Module, cfg: &CompileConfig) -> Result<(), Error> {
    let pm = PassManager::new(&MLIR_CTX);
    if cfg.dump {
        let dump_dir = create_debug_dump_dir().join(MLIR_DUMP_SUBDIR);
        eprintln!("Dumping MLIR to directory `{}`", dump_dir.display());
        pm.enable_ir_printing(&PassIrPrintingOptions {
            before_all: true,
            after_all: true,
            module_scope: false,
            on_change: false,
            on_failure: false,
            flags: OperationPrintingFlags::new(),
            tree_printing_path: dump_dir,
        });
    }

    const INLINER_OPTIONS: &str = "max-iterations=2048";

    // Stage 1: Optimize Qwerty dialect
    // Running the canonicalizer may introduce lambdas, so run it once first
    // before the lambda lifter. Also will optimize ccirc
    pm.add_pass(transform::create_canonicalizer());
    pm.add_pass(qwerty::create_synth_embeds());
    pm.add_pass(qwerty::create_lift_lambdas());
    // Will turn qwerty.call_indirects into qwerty.calls
    pm.add_pass(transform::create_canonicalizer());
    pm.add_pass(transform::create_inliner_with_options(INLINER_OPTIONS));
    // It seems the inliner may not run a final round of canonicalization
    // sometimes, so do it ourselves
    pm.add_pass(transform::create_canonicalizer());
    // Remove any leftover symbols, including ccirc.circuits
    pm.add_pass(transform::create_symbol_dce());

    // Stage 2: Convert to QCirc dialect

    // -only-pred-ones will introduce some lambdas, so lift and inline them too
    pm.add_pass(qwerty::create_only_pred_ones());
    pm.add_pass(qwerty::create_lift_lambdas());
    // Will turn qwerty.call_indirects into qwerty.calls
    pm.add_pass(transform::create_canonicalizer());
    pm.add_pass(transform::create_inliner_with_options(INLINER_OPTIONS));
    pm.add_pass(qwerty::create_qwerty_to_q_circ_conversion());
    // Add canonicalizer pass to prune unused "builtin.unrealized_conversion_cast" ops
    pm.add_pass(transform::create_canonicalizer());

    // Stage 3: Optimize QCirc dialect

    let func_pm = pm.nested_under("func.func");
    func_pm.add_pass(qcirc::create_peephole_optimization());
    match cfg.target {
        Target::Qir(QirProfile::Base) | Target::OpenQasm => {
            func_pm.add_pass(qcirc::create_decompose_multi_control());
            func_pm.add_pass(qcirc::create_peephole_optimization());
            func_pm.add_pass(qcirc::create_barenco_decompose());
        }
        Target::Qir(QirProfile::Unrestricted) => {
            // Maintaining controls (and lowering to __ctl in QIR) makes
            // simulation in qir-runner dramatically faster. So if we are just
            // directly simulating, do only the minimum needed to make the
            // gates valid QIR.
            func_pm.add_pass(qcirc::create_replace_unusual_gates());
        }
    }
    func_pm.add_pass(qcirc::create_peephole_optimization());

    // Stage 4: Convert to QIR
    if matches!(cfg.target, Target::Qir(QirProfile::Base)) {
        pm.add_pass(qcirc::create_base_profile_module_prep());
        let func_pm = pm.nested_under("func.func");
        func_pm.add_pass(qcirc::create_base_profile_func_prep());
    }
    if matches!(cfg.target, Target::Qir(_)) {
        pm.add_pass(qcirc::create_q_circ_to_qir_conversion());
        pm.add_pass(transform::create_canonicalizer());
    }

    pm.run(module)?;

    Ok(())
}

fn create_debug_dump_dir() -> PathBuf {
    let dump_dir = env::current_dir().unwrap().join(QWERTY_DEBUG_DIR);
    fs::create_dir_all(&dump_dir).unwrap();
    dump_dir
}

pub fn translate_to_llvm_ir(
    module: Module<'static>,
    debug: bool,
) -> Result<LLVMModule<'static>, CompileError> {
    let llvm_module = translate_module(module.as_operation(), &LLVM_CTX)
        .ok_or_else(|| CompileError::Message("translation to LLVM IR failed".to_string(), None))?;

    if debug {
        let dump_dir = create_debug_dump_dir();
        let dump_path = dump_dir.join(LLVM_IR_FILENAME);
        eprintln!("Dumping LLVM IR (QIR) to file `{}`", dump_path.display());

        llvm_module
            .print_to_file(dump_path)
            .map_err(|melior_err| CompileError::MLIR(melior_err, None))?;
    }

    Ok(llvm_module)
}

#[derive(Debug)]
pub enum CompileError {
    AST(LowerError),
    MLIR(Error, Option<DebugLoc>),
    Message(String, Option<DebugLoc>),
}

impl From<LowerError> for CompileError {
    fn from(err: LowerError) -> Self {
        CompileError::AST(err)
    }
}

impl From<TypeError> for CompileError {
    fn from(err: TypeError) -> Self {
        CompileError::AST(err.into())
    }
}

pub fn compile_meta_ast(
    prog: &MetaProgram,
    func_name: &str,
    cfg: &CompileConfig,
) -> Result<Module<'static>, CompileError> {
    let plain_ast = prog.lower()?;
    plain_ast.typecheck()?;
    let canon_ast = plain_ast.canonicalize();

    if cfg.dump {
        let dump_dir = create_debug_dump_dir();
        let dump_path = dump_dir.join(QWERTY_AST_FILENAME);
        eprintln!("Dumping Qwerty AST to file `{}`", dump_path.display());

        let dump_str = format!(
            concat!(
                "from qwerty import *\n\n",
                "{}\n",
                "def _run_ast(shots=1024):\n",
                "    return {}(shots=shots)\n\n",
                "if __name__ == '__main__':\n",
                "    histogram(_run_ast())\n"
            ),
            canon_ast.to_python_code(),
            func_name,
        );

        // write out using Display impl
        fs::write(&dump_path, dump_str).unwrap();
    }

    let mut module = ast_program_to_mlir(&canon_ast);

    if cfg.dump {
        let dump_dir = env::current_dir().unwrap().join(QWERTY_DEBUG_DIR);
        fs::create_dir_all(&dump_dir).unwrap();
        let dump_path = dump_dir.join(INIT_MLIR_FILENAME);
        eprintln!(
            "Initial (pre-verification) MLIR be dumped to `{}`",
            dump_path.display()
        );
        module
            .as_operation()
            .print_to_file_with_flags(dump_path, OperationPrintingFlags::new())
            .unwrap();
    }

    assert!(module.as_operation().verify());

    run_passes(&mut module, cfg).map_err(|e| CompileError::MLIR(e, canon_ast.dbg.clone()))?;
    Ok(module)
}

pub fn meta_ast_to_qasm(
    prog: &MetaProgram,
    func_name: &str,
    debug: bool,
) -> Result<String, CompileError> {
    let cfg = CompileConfig {
        target: Target::OpenQasm,
        dump: debug,
    };
    let module = compile_meta_ast(prog, func_name, &cfg)?;
    let sym_table = SymbolTable::new(module.as_operation()).ok_or_else(|| {
        CompileError::Message(
            "Every ModuleOp should have a symbol table".to_string(),
            prog.dbg.clone(),
        )
    })?;
    let func_op = sym_table.lookup(func_name).ok_or_else(|| {
        CompileError::Message(
            format!("Could not find symbol @{}", func_name),
            prog.dbg.clone(),
        )
    })?;

    let print_locs = debug;
    qcirc::generate_qasm((&func_op).into(), print_locs).ok_or_else(|| {
        CompileError::Message("QASM generation failed".to_string(), prog.dbg.clone())
    })
}
