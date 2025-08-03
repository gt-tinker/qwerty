// RUN: ccirc-opt -canonicalize %s | FileCheck %s

// This test verifies that a double ccirc.not is removed by the canonicalizer pass.

// CHECK-LABEL: func.func @double_negation(%arg0: !ccirc.wire) -> !ccirc.wire {
//  CHECK-NEXT:   %0 = ccirc.call @trivial(%arg0) : (!ccirc.wire) -> !ccirc.wire
//  CHECK-NEXT:   ccirc.return %0 : !ccirc.wire
//  CHECK-NEXT: }
func.func @double_negation(%arg0: !ccirc.wire) -> !ccirc.wire {
  %0 = ccirc.not(%arg0) : (!ccirc.wire) -> !ccirc.wire
  %1 = ccirc.not(%0) : (!ccirc.wire) -> !ccirc.wire
  func.return %1 : !ccirc.wire
}