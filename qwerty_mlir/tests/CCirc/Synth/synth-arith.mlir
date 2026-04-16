// RUN: qwerty-opt -convert-ccirc-to-func-arith -canonicalize -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm %s | mlir-runner -e test -entry-point-result=void --shared-libs=%mlir_c_runner_utils | FileCheck --match-full-lines %s

ccirc.circuit @modmul(%y: !ccirc<wire[32]>) irrev {
    %0 = ccirc.modmul 1 1 1 %y : (!ccirc<wire[32]>) -> !ccirc<wire[32]>
    ccirc.return %0 : !ccirc<wire[32]>
}

func.func @check_modmul(%n: i32) -> () {
    %func = ccirc.func_ptr @modmul : (i32) -> (i32)
    %res = func.call_indirect %func(%n) : (i32) -> (i32)
    vector.print %res : i32
    return
}

// run in qwery/build to test this file: python3 ../qwerty_mlir/tests/filecheck_tests.py -k synth_arith
// bin/qwerty-opt -convert-ccirc-to-func-arith -canonicalize -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm ../qwerty_mlir/tests/CCirc/Synth/synth-arith.mlir | mlir-runner -e test -entry-point-result=void --shared-libs=$HOME/bin/llvm21/lib/libmlir_c_runner_utils.so

func.func @test() {
    %a0 = arith.constant 0 : i32
    %a1 = arith.constant 257 : i32
    %a2 = arith.constant 514 : i32
    %a10 = arith.constant 6168 : i32
    %a100 = arith.constant 25700 : i32
    %a103 = arith.constant 74563 : i32
    %aff = arith.constant 65535 : i32
    %acff = arith.constant 131071 : i32

    // 0 + 0 = 0
    // CHECK: 0
    func.call @check_modmul(%a0) : (i32) -> ()

    // 1 + 1 = 2
    // binary: 0 + 10
    // CHECK: 2
    func.call @check_modmul(%a1) : (i32) -> ()

    // 2 + 2 = 4
    // binary: 0 + 100
    // CHECK: 4
    func.call @check_modmul(%a2) : (i32) -> ()

    // 0x18 + 0x18 = 48
    // binary: 0 + 10100
    // CHECK: 48
    func.call @check_modmul(%a10) : (i32) -> ()

    // 100 + 100 = 200
    // binary: 0 + 11001000
    // CHECK: 200
    func.call @check_modmul(%a100) : (i32) -> ()

    // 0x23 + 0x43 = 103
    // CHECK: 103
    func.call @check_modmul(%a103) : (i32) -> ()

    // 0x1 + 0xff + 0xff = 511
    // CHECK: 511
    func.call @check_modmul(%acff) : (i32) -> ()
    return

}
