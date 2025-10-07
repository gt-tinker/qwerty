use super::super::ShotResult;
use dashu::integer::UBig;
use melior::ir::Module;
use qiree_sys::{qiree_create, qiree_destroy};

pub fn run_mlir_module(module: Module, func_name: &str, num_shots: usize) -> Vec<ShotResult> {
    assert_ne!(num_shots, 0);

    unsafe {
        let qiree = qiree_create();
        qiree_destroy(qiree);
    }

    vec![ShotResult {
        bits: UBig::ZERO,
        num_bits: 4,
        count: 1,
    }]
}
