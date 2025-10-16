use crate::compile::{CompileConfig, CompileError, QirProfile, Target, compile_meta_ast};
use dashu::integer::UBig;
use melior::{ExecutionEngine, execution_engine::SymbolFlags, ir::Module};
use qwerty_ast::meta::MetaProgram;
use std::collections::HashMap;

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

fn run_ast(module: Module, func_name: &str, num_shots: usize) -> Vec<ShotResult> {
    assert_ne!(num_shots, 0);

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

pub fn run_meta_ast(
    prog: &MetaProgram,
    func_name: &str,
    num_shots: usize,
    debug: bool,
) -> Result<Vec<ShotResult>, CompileError> {
    let cfg = CompileConfig {
        target: Target::Qir(QirProfile::Unrestricted),
        dump: debug,
    };
    let module = compile_meta_ast(prog, func_name, &cfg)?;
    Ok(run_ast(module, func_name, num_shots))
}
