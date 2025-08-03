// RUN: qwerty-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @double_negation(%arg0: !ccirc.wire) -> !ccirc.wire {
//  CHECK-NEXT:   return %arg0 : !ccirc.wire
//  CHECK-NEXT: }
func.func @double_negation(%arg0: !ccirc.wire) -> !ccirc.wire {
  %0 = ccirc.not(%arg0) : (!ccirc.wire) -> !ccirc.wire
  %1 = ccirc.not(%0) : (!ccirc.wire) -> !ccirc.wire
  return %1 : !ccirc.wire
}
