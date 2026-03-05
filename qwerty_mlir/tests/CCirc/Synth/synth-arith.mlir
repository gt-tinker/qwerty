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

//  run in qwery/build to test this file: python ../qwerty_mlir/tests/filecheck_tests.py -k synth_arith


func.func @test() {
    %cst0 = arith.constant 0 : i32
    %cst1 = arith.constant 1 : i32
    %cst2 = arith.constant 2 : i32
    %cst3 = arith.constant 3 : i32
    %cst4 = arith.constant 4 : i32
    %cst5 = arith.constant 5 : i32
    %cst6 = arith.constant 6 : i32
    %cst7 = arith.constant 7 : i32

    //      
    // tests for fullAddr1:
    // add  a + b + cin
    //

    // synthModMul(0x00000000) == 0x00000000
    // CHECK: 0
    func.call @check_modmul(%cst0) : (i32) -> ()

    // synthModMul(0x00000001) == 0x00000002
    // CHECK: 2
    func.call @check_modmul(%cst1) : (i32) -> ()

    // synthModMul(0x00000002) == 0x00000002
    // CHECK: 2
    func.call @check_modmul(%cst2) : (i32) -> ()

    // synthModMul(0x00000003) == 0x00000001
    // CHECK: 1
    func.call @check_modmul(%cst3) : (i32) -> ()

    // synthModMul(0x00000004) == 0x00000002
    // CHECK: 2
    func.call @check_modmul(%cst4) : (i32) -> ()

    // synthModMul(0x00000005) == 0x00000001
    // CHECK: 1
    func.call @check_modmul(%cst5) : (i32) -> ()

    // synthModMul(0x00000006) == 0x00000001
    // CHECK: 1
    func.call @check_modmul(%cst6) : (i32) -> ()

    // synthModMul(0x00000007) == 0x00000003
    // CHECK: 3
    func.call @check_modmul(%cst7) : (i32) -> ()


    return




    
    


}
