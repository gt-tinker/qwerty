use crate::mlir::{ctx::MLIR_CTX, lower_prog_stmt::ast_program_to_mlir};
use dashu::integer::UBig;
use melior::{
    dialect::{qcirc, qwerty},
    execution_engine::SymbolFlags,
    ir::{operation::OperationPrintingFlags, Module},
    pass::{transform, PassIrPrintingOptions, PassManager},
    Error, ExecutionEngine,
};
use qwerty_ast::ast::Program;
use std::{collections::HashMap, env};

struct RunPassesConfig {
    decompose_multi_ctrl: bool,
    to_base_profile: bool,
    dump: bool,
}

fn run_passes(module: &mut Module, cfg: RunPassesConfig) -> Result<(), Error> {
    let pm = PassManager::new(&MLIR_CTX);
    if cfg.dump {
        let dump_dir = env::current_dir().unwrap().join("mlir-dumps");
        eprintln!(
            "MLIR files will be dumped to directory `{}`",
            dump_dir.display()
        );
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

    // Stage 1: Optimize Qwerty dialect

    // Running the canonicalizer may introduce lambdas, so run it once first
    // before the lambda lifter. Also will optimize ccirc
    pm.add_pass(transform::create_canonicalizer());
    pm.add_pass(qwerty::create_synth_embeds());
    pm.add_pass(qwerty::create_lift_lambdas());
    // Will turn qwerty.call_indirects into qwerty.calls
    pm.add_pass(transform::create_canonicalizer());
    pm.add_pass(transform::create_inliner());
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
    pm.add_pass(transform::create_inliner());
    pm.add_pass(qwerty::create_qwerty_to_q_circ_conversion());
    // Add canonicalizer pass to prune unused "builtin.unrealized_conversion_cast" ops
    pm.add_pass(transform::create_canonicalizer());

    // Stage 3: Optimize QCirc dialect

    let func_pm = pm.nested_under("func.func");
    func_pm.add_pass(qcirc::create_peephole_optimization());
    if cfg.decompose_multi_ctrl {
        func_pm.add_pass(qcirc::create_decompose_multi_control());
        func_pm.add_pass(qcirc::create_peephole_optimization());
        func_pm.add_pass(qcirc::create_replace_non_qasm_gates());
    }

    // Stage 4: Convert to QIR
    pm.add_pass(qcirc::create_replace_non_qir_gates());
    if cfg.to_base_profile {
        pm.add_pass(qcirc::create_base_profile_module_prep());
        let func_pm = pm.nested_under("func.func");
        func_pm.add_pass(qcirc::create_base_profile_func_prep());
    }
    pm.add_pass(qcirc::create_q_circ_to_qir_conversion());
    pm.add_pass(transform::create_canonicalizer());

    pm.run(module)?;

    Ok(())
}

pub struct ShotResult {
    pub bits: UBig,
    pub num_bits: usize,
    pub count: usize,
}

macro_rules! qir_symbol {
    ($func_name:ident) => {
        (
            stringify!($func_name),
            qir_backend::$func_name as *mut (),
            SymbolFlags::CALLABLE,
        )
    };
}

pub fn run_ast(prog: &Program, func_name: &str, num_shots: usize, debug: bool) -> Vec<ShotResult> {
    assert_ne!(num_shots, 0);

    let mut module = ast_program_to_mlir(prog);
    let cfg = RunPassesConfig {
        decompose_multi_ctrl: false,
        to_base_profile: false,
        dump: debug,
    };
    run_passes(&mut module, cfg).unwrap();

    let exec = ExecutionEngine::new(&module, 3, &[], false);

    unsafe {
        exec.register_symbols(&[
            qir_symbol!(__quantum__rt__initialize),
            qir_symbol!(__quantum__qis__x__body),
            qir_symbol!(__quantum__qis__y__body),
            qir_symbol!(__quantum__qis__z__body),
            qir_symbol!(__quantum__qis__h__body),
            qir_symbol!(__quantum__qis__rx__body),
            qir_symbol!(__quantum__qis__ry__body),
            qir_symbol!(__quantum__qis__rz__body),
            qir_symbol!(__quantum__qis__s__body),
            qir_symbol!(__quantum__qis__s__adj),
            qir_symbol!(__quantum__qis__t__body),
            qir_symbol!(__quantum__qis__t__adj),
            qir_symbol!(__quantum__qis__cx__body),
            qir_symbol!(__quantum__qis__cy__body),
            qir_symbol!(__quantum__qis__cz__body),
            qir_symbol!(__quantum__qis__ccx__body),
            qir_symbol!(__quantum__qis__x__ctl),
            qir_symbol!(__quantum__qis__y__ctl),
            qir_symbol!(__quantum__qis__z__ctl),
            qir_symbol!(__quantum__qis__h__ctl),
            qir_symbol!(__quantum__qis__rx__ctl),
            qir_symbol!(__quantum__qis__ry__ctl),
            qir_symbol!(__quantum__qis__rz__ctl),
            qir_symbol!(__quantum__qis__s__ctl),
            qir_symbol!(__quantum__qis__s__ctladj),
            qir_symbol!(__quantum__qis__t__ctl),
            qir_symbol!(__quantum__qis__t__ctladj),
            qir_symbol!(__quantum__qis__m__body),
            qir_symbol!(__quantum__qis__reset__body),
            qir_symbol!(__quantum__rt__result_get_one),
            qir_symbol!(__quantum__rt__result_equal),
            qir_symbol!(__quantum__rt__qubit_allocate),
            qir_symbol!(__quantum__rt__qubit_release),
            qir_symbol!(__quantum__rt__array_create_1d),
            qir_symbol!(__quantum__rt__array_copy),
            qir_symbol!(__quantum__rt__array_update_reference_count),
            qir_symbol!(__quantum__rt__array_update_alias_count),
            qir_symbol!(__quantum__rt__array_get_element_ptr_1d),
            qir_symbol!(__quantum__rt__array_get_size_1d),
            qir_symbol!(__quantum__rt__tuple_create),
            qir_symbol!(__quantum__rt__tuple_update_reference_count),
            qir_symbol!(__quantum__rt__tuple_update_alias_count),
            qir_symbol!(__quantum__rt__callable_create),
            qir_symbol!(__quantum__rt__callable_copy),
            qir_symbol!(__quantum__rt__callable_invoke),
            qir_symbol!(__quantum__rt__callable_make_adjoint),
            qir_symbol!(__quantum__rt__callable_make_controlled),
            qir_symbol!(__quantum__rt__callable_update_reference_count),
            qir_symbol!(__quantum__rt__callable_update_alias_count),
            qir_symbol!(__quantum__rt__capture_update_reference_count),
            qir_symbol!(__quantum__rt__capture_update_alias_count),
        ]);
        qir_backend::__quantum__rt__initialize(std::ptr::null::<u8>() as *mut _);
    }

    let mut counts: HashMap<UBig, usize> = HashMap::new();
    let mut num_bits_ret = None;

    for _ in 0..num_shots {
        // TODO: check that the function really takes no arguments and returns a bitbundle
        let (bits, num_bits_qir) = unsafe {
            let mut result: *const qir_backend::QirArray = std::ptr::null();
            exec.invoke_packed(func_name, &mut [&raw mut result as *mut ()])
                .unwrap();

            let num_bits = qir_backend::__quantum__rt__array_get_size_1d(result);
            let mut bits_ubig = UBig::ZERO;
            for i in 0..num_bits {
                bits_ubig <<= 1usize;
                if *qir_backend::__quantum__rt__array_get_element_ptr_1d(result, i) != 0 {
                    bits_ubig |= 1usize;
                }
            }
            qir_backend::__quantum__rt__array_update_reference_count(result, -1);
            (bits_ubig, num_bits)
        };

        if let Some(nbits) = num_bits_ret {
            assert_eq!(nbits, num_bits_qir);
        } else {
            num_bits_ret = Some(num_bits_qir);
        }

        counts
            .entry(bits)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    let num_bits: usize = num_bits_ret
        .expect("There should be at least one shot")
        .try_into()
        .unwrap();

    counts
        .iter()
        .map(|(bits, count)| ShotResult {
            bits: bits.clone(),
            num_bits,
            count: *count,
        })
        .collect()
}
