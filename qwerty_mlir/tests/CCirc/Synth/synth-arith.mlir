// RUN: qwerty-opt -convert-ccirc-to-func-arith -canonicalize -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm %s | mlir-runner -e test -entry-point-result=void --shared-libs=%mlir_c_runner_utils | FileCheck --match-full-lines %s

ccirc.circuit @add(%a: !ccirc<wire[8]>, %b: !ccirc<wire[8]>) irrev {
    %0 = ccirc.add(%a, %b) : (!ccirc<wire[8]>, !ccirc<wire[8]>) -> !ccirc<wire[8]>
    ccirc.return %0 : !ccirc<wire[8]>
}

func.func @check_add(%a: i8, %b: i8) -> () {
    %func = ccirc.func_ptr @add : (i8, i8) -> (i8)
    %res = func.call_indirect %func(%a, %b) : (i8, i8) -> (i8)
    vector.print %res : i8
    return
}

func.func @test() {
    %c0 = arith.constant 0 : i8
    %c1 = arith.constant 1 : i8
    %c2 = arith.constant 2 : i8
    %c24 = arith.constant 24 : i8
    %c100 = arith.constant 100 : i8
    %c35 = arith.constant 35 : i8
    %c67 = arith.constant 67 : i8
    %c255 = arith.constant 255 : i8

    // 0 + 0 = 0
    // CHECK: 0
    func.call @check_add(%c0, %c0) : (i8, i8) -> ()

    // 1 + 1 = 2
    // CHECK: 2
    func.call @check_add(%c1, %c1) : (i8, i8) -> ()

    // 2 + 2 = 4
    // CHECK: 4
    func.call @check_add(%c2, %c2) : (i8, i8) -> ()

    // 24 + 24 = 48
    // CHECK: 48
    func.call @check_add(%c24, %c24) : (i8, i8) -> ()

    // 100 + 100 = 200
    // Will be printed as signed int.
    // Note that 200 | -1 << 8 == -56.
    // CHECK: -56
    func.call @check_add(%c100, %c100) : (i8, i8) -> ()

    // 35 + 67 = 102
    // CHECK: 102
    func.call @check_add(%c35, %c67) : (i8, i8) -> ()

    // 255 + 255 = 254
    // Will be printed as signed int.
    // Note that 254 | -1 << 8 == -2.
    // CHECK: -2
    func.call @check_add(%c255, %c255) : (i8, i8) -> ()
    return

}
