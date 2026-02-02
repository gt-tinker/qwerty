// RUN: qwerty-opt -convert-ccirc-to-func-arith -canonicalize -convert-vector-to-llvm -convert-func-to-llvm -convert-arith-to-llvm %s | mlir-runner -e test -entry-point-result=void --shared-libs=%mlir_c_runner_utils | FileCheck --match-full-lines %s

ccirc.circuit @rotl(%n: !ccirc<wire[32]>, %k: !ccirc<wire[5]>) irrev {
    %0 = ccirc.rotl(%n, %k) : (!ccirc<wire[32]>, !ccirc<wire[5]>) -> !ccirc<wire[32]>
    ccirc.return %0 : !ccirc<wire[32]>
}

func.func @check_rotl(%n: i32, %k: i5) -> () {
    %func = ccirc.func_ptr @rotl : (i32, i5) -> (i32)
    %res = func.call_indirect %func(%n, %k) : (i32, i5) -> (i32)
    vector.print %res : i32
    return
}

ccirc.circuit @rotr(%n: !ccirc<wire[32]>, %k: !ccirc<wire[5]>) irrev {
    %0 = ccirc.rotr(%n, %k) : (!ccirc<wire[32]>, !ccirc<wire[5]>) -> !ccirc<wire[32]>
    ccirc.return %0 : !ccirc<wire[32]>
}

func.func @check_rotr(%n: i32, %k: i5) -> () {
    %func = ccirc.func_ptr @rotr : (i32, i5) -> (i32)
    %res = func.call_indirect %func(%n, %k) : (i32, i5) -> (i32)
    vector.print %res : i32
    return
}

func.func @test() {
    %cst1_32 = arith.constant 1 : i32
    %cst80_00_00_00_32 = arith.constant -2147483648 : i32
    %cst-1_32 = arith.constant -1 : i32
    %cstEE_EE_EE_EE_32 = arith.constant -286331154 : i32

    %cst1_5 = arith.constant 1 : i5
    %cst12_5 = arith.constant 12 : i5
    %cst13_5 = arith.constant 13 : i5

    //
    // Rotate Left
    //

    // rotl(0x00000001, 1) == 0x00000002
    // CHECK: 2
    func.call @check_rotl(%cst1_32, %cst1_5) : (i32, i5) -> ()

    // rotl(0x80000000, 1) == 0x00000001
    // CHECK-NEXT: 1
    func.call @check_rotl(%cst80_00_00_00_32, %cst1_5) : (i32, i5) -> ()

    // rotl(0xFFFFFFFF, 13) == 0xFFFFFFFF
    // CHECK-NEXT: -1
    func.call @check_rotl(%cst-1_32, %cst13_5) : (i32, i5) -> ()

    // rotl(0xEEEEEEEE, 13) == 0xDDDDDDDD
    // CHECK-NEXT: -572662307
    func.call @check_rotl(%cstEE_EE_EE_EE_32, %cst13_5) : (i32, i5) -> ()

    // rotl(0xEEEEEEEE, 12) == 0xEEEEEEEE
    // CHECK-NEXT: -286331154
    func.call @check_rotl(%cstEE_EE_EE_EE_32, %cst12_5) : (i32, i5) -> ()

    //
    // Rotate Right
    //

    // rotr(0x00000001, 1) == 0x80000000
    // CHECK-NEXT: -2147483648
    func.call @check_rotr(%cst1_32, %cst1_5) : (i32, i5) -> ()

    // rotr(0x80000000, 1) == 0x40000000
    // CHECK-NEXT: 1073741824
    func.call @check_rotr(%cst80_00_00_00_32, %cst1_5) : (i32, i5) -> ()

    // rotr(0xFFFFFFFF, 13) == 0xFFFFFFFF
    // CHECK-NEXT: -1
    func.call @check_rotr(%cst-1_32, %cst13_5) : (i32, i5) -> ()

    // rotr(0xEEEEEEEE, 13) == 0x77777777
    // CHECK-NEXT: 2004318071
    func.call @check_rotr(%cstEE_EE_EE_EE_32, %cst13_5) : (i32, i5) -> ()

    // rotr(0xEEEEEEEE, 12) == 0xEEEEEEEE
    // CHECK-NEXT: -286331154
    func.call @check_rotr(%cstEE_EE_EE_EE_32, %cst12_5) : (i32, i5) -> ()

    return
}
