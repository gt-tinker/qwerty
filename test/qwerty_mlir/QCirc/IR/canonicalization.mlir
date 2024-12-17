// RUN: qwerty-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @chained_calc(%arg0: f64) -> f64 {
//  CHECK-NEXT:   %0 = qcirc.calc(%arg0 as %arg1) : (f64) -> f64 {
//  CHECK-NEXT:     %cst = arith.constant 4.200000e+00 : f64
//  CHECK-NEXT:     %cst_0 = arith.constant 3.140000e+00 : f64
//  CHECK-NEXT:     %1 = arith.addf %arg1, %cst_0 : f64
//  CHECK-NEXT:     %2 = arith.addf %1, %cst : f64
//  CHECK-NEXT:     qcirc.calc_yield(%2) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %0 : f64
//  CHECK-NEXT: }
func.func @chained_calc(%arg0: f64) -> f64 {
  %0 = qcirc.calc(%arg0 as %arg1) : (f64) -> f64 {
    %1 = arith.constant 3.14 : f64
    %2 = arith.addf %arg1, %1 : f64
    qcirc.calc_yield(%2) : f64
  }
  %3 = qcirc.calc(%0 as %arg2) : (f64) -> f64 {
    %4 = arith.constant 4.20 : f64
    %5 = arith.addf %4, %arg2 : f64
    qcirc.calc_yield(%5) : f64
  }
  return %3 : f64
}

// CHECK-LABEL: func.func @chained_calc_int(%arg0: i64) -> i64 {
//  CHECK-NEXT:   %0 = qcirc.calc(%arg0 as %arg1) : (i64) -> i64 {
//  CHECK-NEXT:     %c73_i64 = arith.constant 73 : i64
//  CHECK-NEXT:     %1 = arith.addi %arg1, %c73_i64 : i64
//  CHECK-NEXT:     qcirc.calc_yield(%1) : i64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %0 : i64
//  CHECK-NEXT: }
func.func @chained_calc_int(%arg0: i64) -> i64 {
  %0 = qcirc.calc(%arg0 as %arg1) : (i64) -> i64 {
    %1 = arith.constant 4 : i64
    %2 = arith.addi %arg1, %1 : i64
    qcirc.calc_yield(%2) : i64
  }
  %3 = qcirc.calc(%0 as %arg2) : (i64) -> i64 {
    %4 = arith.constant 69 : i64
    %5 = arith.addi %4, %arg2 : i64
    qcirc.calc_yield(%5) : i64
  }
  return %3 : i64
}
