// RUN: qwerty-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: ccirc.circuit @double_negation(%arg0: !ccirc<wire[3]>) irrev {
//  CHECK-NEXT:   ccirc.return %arg0 : !ccirc<wire[3]>
//  CHECK-NEXT: }
ccirc.circuit @double_negation(%arg0: !ccirc<wire[3]>) irrev {
  %0 = ccirc.not(%arg0) : (!ccirc<wire[3]>) -> !ccirc<wire[3]>
  %1 = ccirc.not(%0) : (!ccirc<wire[3]>) -> !ccirc<wire[3]>
  ccirc.return %1 : !ccirc<wire[3]>
}

// CHECK-LABEL: ccirc.circuit @pack_unpack(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>, %arg2: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   ccirc.return %arg0, %arg1, %arg2 : !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @pack_unpack(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>, %arg2: !ccirc<wire[1]>) irrev {
  %0 = ccirc.wirepack(%arg0, %arg1, %arg2) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
  %1:3 = ccirc.wireunpack %0 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  ccirc.return %1#0, %1#1, %1#2 : !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>
}

// CHECK-LABEL: ccirc.circuit @pack_unpack_trivial(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>, %arg2: !ccirc<wire[1]>) irrev {
//  CHECK-NEXT:   ccirc.return %arg0, %arg1, %arg2 : !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>
//  CHECK-NEXT: }
ccirc.circuit @pack_unpack_trivial(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[1]>, %arg2: !ccirc<wire[1]>) irrev {
  %0 = ccirc.wirepack(%arg0, %arg1, %arg2) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
  %1:3 = ccirc.wireunpack %0 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %2 = ccirc.wirepack(%1#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.wirepack(%1#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %4 = ccirc.wirepack(%1#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %2, %3, %4 : !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>
}
