// RUN: qwerty-opt -convert-ccirc-to-func-arith -canonicalize -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm %s | mlir-runner -e test -entry-point-result=void --shared-libs=%mlir_c_runner_utils | FileCheck --match-full-lines %s

ccirc.circuit @bitwise_not(%arg0: !ccirc<wire[32]>) irrev {
    %0 = ccirc.not(%arg0) : (!ccirc<wire[32]>) -> !ccirc<wire[32]>
    ccirc.return %0 : !ccirc<wire[32]>
}

func.func @check_bitwise_not(%val : i32) -> () {
    %func = ccirc.func_ptr @bitwise_not : (i32) -> (i32)
    %res = func.call_indirect %func(%val) : (i32) -> (i32)
    vector.print %res : i32
    return
}

func.func @test() {
    %cst0 = arith.constant 0 : i32
    %cst-1 = arith.constant -1 : i32

    // CHECK: -1
    func.call @check_bitwise_not(%cst0) : (i32) -> ()

    // CHECK-NEXT: 0
    func.call @check_bitwise_not(%cst-1) : (i32) -> ()
    return
}
