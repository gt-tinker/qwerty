// RUN: qwerty-opt -inline-adj %s | FileCheck %s

// CHECK-LABEL: func.func @trivial(%arg0: !qcirc.qubit) -> !qcirc.qubit {
//  CHECK-NEXT:   return %arg0 : !qcirc.qubit
//  CHECK-NEXT: }
func.func @trivial(%arg0: !qcirc.qubit) -> !qcirc.qubit {
  %0 = qcirc.adj(%arg0 as %q) : (!qcirc.qubit) -> !qcirc.qubit {
    qcirc.yield(%q) : !qcirc.qubit
  }
  return %0 : !qcirc.qubit
}

// CHECK-LABEL: func.func @hermitian_gate(%arg0: !qcirc.qubit) -> !qcirc.qubit {
//  CHECK-NEXT:   %result = qcirc.gate1q[]:X %arg0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   return %result : !qcirc.qubit
//  CHECK-NEXT: }
func.func @hermitian_gate(%arg0: !qcirc.qubit) -> !qcirc.qubit {
  %0 = qcirc.adj(%arg0 as %q) : (!qcirc.qubit) -> !qcirc.qubit {
    %result = qcirc.gate1q[]:X %q : (!qcirc.qubit) -> !qcirc.qubit
    qcirc.yield(%result) : !qcirc.qubit
  }
  return %0 : !qcirc.qubit
}

// CHECK-LABEL: func.func @phase(%arg0: !qcirc.qubit) -> !qcirc.qubit {
//  CHECK-NEXT:   %0 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant 1.230000e+00 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %1 = qcirc.calc(%0 as %arg1) : (f64) -> f64 {
//  CHECK-NEXT:     %2 = arith.negf %arg1 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%2) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %result = qcirc.gate1q1p[]:P(%1) %arg0 : (f64, !qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   return %result : !qcirc.qubit
//  CHECK-NEXT: }
func.func @phase(%arg0: !qcirc.qubit) -> !qcirc.qubit {
  %0 = qcirc.adj(%arg0 as %q) : (!qcirc.qubit) -> !qcirc.qubit {
    %calc = qcirc.calc() : () -> f64 {
        %1 = arith.constant 1.23 : f64
        qcirc.calc_yield(%1) : f64
    }
    %result = qcirc.gate1q1p[]:P(%calc) %q : (f64, !qcirc.qubit) -> !qcirc.qubit
    qcirc.yield(%result) : !qcirc.qubit
  }
  return %0 : !qcirc.qubit
}

// CHECK-LABEL: func.func @one_qubit_gate_seq(%arg0: !qcirc.qubit) -> !qcirc.qubit {
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sxdg %arg0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:Tdg %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:S %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   return %result_1 : !qcirc.qubit
//  CHECK-NEXT: }
func.func @one_qubit_gate_seq(%arg0: !qcirc.qubit) -> !qcirc.qubit {
  %0 = qcirc.adj(%arg0 as %q) : (!qcirc.qubit) -> !qcirc.qubit {
    %result = qcirc.gate1q[]:Sdg %q : (!qcirc.qubit) -> !qcirc.qubit
    %result_1 = qcirc.gate1q[]:T %result : (!qcirc.qubit) -> !qcirc.qubit
    %result_2 = qcirc.gate1q[]:Sx %result_1 : (!qcirc.qubit) -> !qcirc.qubit
    qcirc.yield(%result_2) : !qcirc.qubit
  }
  return %0 : !qcirc.qubit
}

// CHECK-LABEL: func.func @one_qubit_gate_seq_w_passthru(%arg0: !qcirc.qubit, %arg1: !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit) {
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sxdg %arg1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:Tdg %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:S %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   return %result_1, %arg0 : !qcirc.qubit, !qcirc.qubit
//  CHECK-NEXT: }
func.func @one_qubit_gate_seq_w_passthru(%arg0: !qcirc.qubit, %arg1: !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit) {
  %0:2 = qcirc.adj(%arg0 as %q0, %arg1 as %q1) : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit) {
    %result = qcirc.gate1q[]:Sdg %q0 : (!qcirc.qubit) -> !qcirc.qubit
    %result_1 = qcirc.gate1q[]:T %result : (!qcirc.qubit) -> !qcirc.qubit
    %result_2 = qcirc.gate1q[]:Sx %result_1 : (!qcirc.qubit) -> !qcirc.qubit
    qcirc.yield(%q1, %result_2) : !qcirc.qubit, !qcirc.qubit
  }
  return %0#0, %0#1 : !qcirc.qubit, !qcirc.qubit
}

// CHECK-LABEL: func.func @ancilla(%arg0: !qcirc.qubit, %arg1: !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit) {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %leftResult, %rightResult = qcirc.gate2q[]:Swap %arg0, %arg1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%leftResult]:X %0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%rightResult]:X %1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %leftResult_2, %rightResult_3 = qcirc.gate2q[]:Swap %controlResults, %controlResults_0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_4, %result_5 = qcirc.gate1q[%leftResult_2]:X %result_1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_6, %result_7 = qcirc.gate1q[%rightResult_3]:X %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_5 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_7 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   return %controlResults_4, %controlResults_6 : !qcirc.qubit, !qcirc.qubit
//  CHECK-NEXT: }
func.func @ancilla(%arg0: !qcirc.qubit, %arg1: !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit) {
  %0:2 = qcirc.adj(%arg0 as %q0, %arg1 as %q1) : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit) {
    %0 = qcirc.qalloc : () -> !qcirc.qubit
    %1:2 = qcirc.gate1q[%q0]:X %0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %2 = qcirc.qalloc : () -> !qcirc.qubit
    %3:2 = qcirc.gate1q[%q1]:X %2 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %4:2 = qcirc.gate2q[]:Swap %1#0, %3#0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %5:2 = qcirc.gate1q[%4#0]:X %3#1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %6:2 = qcirc.gate1q[%4#1]:X %1#1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %7:2 = qcirc.gate2q[]:Swap %5#0, %6#0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    qcirc.qfreez %6#1 : (!qcirc.qubit) -> ()
    qcirc.qfreez %5#1 : (!qcirc.qubit) -> ()
    qcirc.yield(%7#0, %7#1) : !qcirc.qubit, !qcirc.qubit
  }
  return %0#0, %0#1 : !qcirc.qubit, !qcirc.qubit
}
