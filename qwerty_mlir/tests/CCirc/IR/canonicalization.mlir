// RUN: qwerty-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @double_negation(%arg0: !ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]> {
//  CHECK-NEXT:   return %arg0 : !ccirc<wirebundle[3]>
//  CHECK-NEXT: }
func.func @double_negation(%arg0: !ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]> {
  %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
  %1 = ccirc.not(%0) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
  return %1 : !ccirc<wirebundle[3]>
}
