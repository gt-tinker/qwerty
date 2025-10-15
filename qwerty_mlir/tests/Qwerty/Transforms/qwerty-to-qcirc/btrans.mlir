// RUN: qwerty-opt -convert-qwerty-to-qcirc -canonicalize -peephole-optimization %s | FileCheck %s

// CHECK-LABEL: func.func @hadamard() -> !qcirc<array<i1>[1]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:H %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %1 = qcirc.arrpack(%measResult) : (i1) -> !qcirc<array<i1>[1]>
//  CHECK-NEXT:   return %1 : !qcirc<array<i1>[1]>
//  CHECK-NEXT: }
qwerty.func @hadamard[]() irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.qbtrans %0 by {std:Z[1]} >> {std:X[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %2 = qwerty.qbmeas %1 by {std:Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  qwerty.return %2 : !qwerty<bitbundle[1]>
}

// CHECK-LABEL: func.func @withCtrl() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sdg %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result_1 = qcirc.gate1q[%result_0]:H %1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:H %controlResults : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:S %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_3) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_4, %measResult_5 = qcirc.measure(%result_1) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_4 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_5) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @withCtrl[]() irrev-> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.qbtrans %0 by {list:{"|j>"}, std:Z[1]} >> {list:{"|j>"}, std:X[1]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std:Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @withCtrlPhase() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sdg %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:Z %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result_2 = qcirc.gate1q[%result_1]:H %1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:H %controlResults : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_4 = qcirc.gate1q[]:S %result_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_4) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_5, %measResult_6 = qcirc.measure(%result_2) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_5 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_6) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @withCtrlPhase[]() irrev-> !qwerty<bitbundle[2]> {
  %cst = arith.constant 3.1415926535897931 : f64
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.qbtrans %0 by {list:{"|j>"}, std:Z[1]} >> {list:{exp(i*theta)*"|j>"}, std:X[1]} phases (%cst) : (f64, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std:Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @withCtrlXPhase() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sdg %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result_1 = qcirc.gate1q[%result_0]:X %1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:Z %controlResults : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_3, %result_4 = qcirc.gate1q[%result_2]:H %result_1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:H %controlResults_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:S %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_6) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_7, %measResult_8 = qcirc.measure(%result_4) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_7 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_8) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @withCtrlXPhase[]() irrev-> !qwerty<bitbundle[2]> {
  %cst = arith.constant 3.1415926535897931 : f64
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.qbtrans %0 by {list:{"|j>"}, list:{"|1>","|0>"}} >> {list:{exp(i*theta)*"|j>"}, std:X[1]} phases (%cst) : (f64, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std:Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @qft() -> !qcirc<array<i1>[3]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:H %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result_0 = qcirc.gate1q[%1]:S %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_1, %result_2 = qcirc.gate1q[%2]:T %result_0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:H %controlResults : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_4, %result_5 = qcirc.gate1q[%controlResults_1]:S %result_3 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:H %controlResults_4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_6) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_7, %measResult_8 = qcirc.measure(%result_5) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_7 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_9, %measResult_10 = qcirc.measure(%result_2) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_9 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %3 = qcirc.arrpack(%measResult, %measResult_8, %measResult_10) : (i1, i1, i1) -> !qcirc<array<i1>[3]>
//  CHECK-NEXT:   return %3 : !qcirc<array<i1>[3]>
//  CHECK-NEXT: }
qwerty.func @qft[]() irrev-> !qwerty<bitbundle[3]> {
  %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %2 = qwerty.qbunpack %0 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %3 = qwerty.qbunpack %1 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %4 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %5 = qwerty.qbunpack %4 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %6 = qwerty.qbpack(%2, %3, %5) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
  %7 = qwerty.qbtrans %6 by {std: Z[3]} >> {revolve: {revolve: {std: X[1]} by {"|0>", "|1>"}} by {"|0>", "|1>"}} : (!qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
  %8:3 = qwerty.qbunpack %7 : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
  %9 = qwerty.qbpack(%8#0) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
  %10 = qwerty.qbmeas %9 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %11 = qwerty.bitunpack %10 : (!qwerty<bitbundle[1]>) -> i1
  %12 = qwerty.qbpack(%8#1) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
  %13 = qwerty.qbmeas %12 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %14 = qwerty.bitunpack %13 : (!qwerty<bitbundle[1]>) -> i1
  %15 = qwerty.qbpack(%8#2) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
  %16 = qwerty.qbmeas %15 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %17 = qwerty.bitunpack %16 : (!qwerty<bitbundle[1]>) -> i1
  %18 = qwerty.bitpack(%11, %14, %17) : (i1, i1, i1) -> !qwerty<bitbundle[3]>
  qwerty.return %18 : !qwerty<bitbundle[3]>
}

// TODO: Fourier identity

// CHECK-LABEL: func.func @pm_identity() -> !qcirc<array<i1>[3]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%0) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_0, %measResult_1 = qcirc.measure(%1) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_0 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_2, %measResult_3 = qcirc.measure(%2) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_2 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %3 = qcirc.arrpack(%measResult, %measResult_1, %measResult_3) : (i1, i1, i1) -> !qcirc<array<i1>[3]>
//  CHECK-NEXT:   return %3 : !qcirc<array<i1>[3]>
//  CHECK-NEXT: }
qwerty.func @pm_identity[]() irrev-> !qwerty<bitbundle[3]> {
  %0 = qwerty.qbprep Z<PLUS>[3] : () -> !qwerty<qbundle[3]>
  %1 = qwerty.qbtrans %0 by {std:X[3]} >> {std:X[3]} : (!qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
  %2 = qwerty.qbmeas %1 by {std:Z[3]} : !qwerty<qbundle[3]> -> !qwerty<bitbundle[3]>
  qwerty.return %2 : !qwerty<bitbundle[3]>
}

// CHECK-LABEL: func.func @swap() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%1]:X %0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%result]:X %controlResults : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2, %result_3 = qcirc.gate1q[%result_1]:X %controlResults_0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_3) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_4, %measResult_5 = qcirc.measure(%controlResults_2) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_4 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_5) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @swap[]() irrev-> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.qbtrans %0 by {list:{"|01>","|10>"}} >> {list:{"|10>","|01>"}} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std:Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @cnot() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:X %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result_0 = qcirc.gate1q[%result]:X %1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%controlResults) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_1, %measResult_2 = qcirc.measure(%result_0) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_1 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_2) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @cnot[]() irrev-> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<MINUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %2 = qwerty.qbunpack %0 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %3 = qwerty.qbunpack %1 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %4 = qwerty.qbpack(%2, %3) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
  %5 = qwerty.qbtrans %4 by {list:{"|00>", "|01>", "|10>", "|11>"}} >> {list:{"|00>", "|01>", "|11>", "|10>"}} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %6 = qwerty.qbmeas %5 by {std: Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %6 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @to_bell() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%1]:Z %0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:H %controlResults : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_1, %result_2 = qcirc.gate1q[%result_0]:X %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_2) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_3, %measResult_4 = qcirc.measure(%controlResults_1) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_3 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_4) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @to_bell[]() irrev-> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.qbtrans %0 by {std:Z[2]} >> {std:BELL[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std:Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @small_perm() -> !qcirc<array<i1>[2]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%1]:X %0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%result]:X %controlResults : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2, %result_3 = qcirc.gate1q[%result_1]:X %controlResults_0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_4 = qcirc.gate1q[]:X %result_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:X %controlResults_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_4) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_6, %measResult_7 = qcirc.measure(%result_5) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_6 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %2 = qcirc.arrpack(%measResult, %measResult_7) : (i1, i1) -> !qcirc<array<i1>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<i1>[2]>
//  CHECK-NEXT: }
qwerty.func @small_perm[]() irrev-> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.qbtrans %0 by {list:{"|00>", "|11>"}} >> {list:{"|11>", "|00>"}} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std: Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: func.func @big_perm() -> !qcirc<array<i1>[11]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %3 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %4 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %5 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %6 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %7 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %8 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %9 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %10 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %11 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:X %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:X %1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:X %2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:X %3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:X %4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_4 = qcirc.gate1q[]:X %5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:X %6 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:X %7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_7 = qcirc.gate1q[]:X %8 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_8 = qcirc.gate1q[]:X %9 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:X %10 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults:11, %result_10 = qcirc.gate1q[%result, %result_0, %result_1, %result_2, %result_3, %result_4, %result_5, %result_6, %result_7, %result_8, %result_9]:X %11 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_11 = qcirc.gate1q[]:X %controlResults#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_12 = qcirc.gate1q[]:X %controlResults#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_13 = qcirc.gate1q[]:X %controlResults#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_14 = qcirc.gate1q[]:X %controlResults#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_15 = qcirc.gate1q[]:X %controlResults#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_16 = qcirc.gate1q[]:X %controlResults#5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_17 = qcirc.gate1q[]:X %controlResults#6 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_18 = qcirc.gate1q[]:X %controlResults#7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_19 = qcirc.gate1q[]:X %controlResults#8 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_20 = qcirc.gate1q[]:X %controlResults#9 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_21 = qcirc.gate1q[]:X %controlResults#10 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_22:11, %result_23 = qcirc.gate1q[%result_11, %result_12, %result_13, %result_14, %result_15, %result_16, %result_17, %result_18, %result_19, %result_20, %result_21]:X %result_10 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_24, %result_25 = qcirc.gate1q[%result_23]:X %controlResults_22#0 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_26, %result_27 = qcirc.gate1q[%controlResults_24]:X %controlResults_22#1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_28, %result_29 = qcirc.gate1q[%controlResults_26]:X %controlResults_22#2 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_30, %result_31 = qcirc.gate1q[%controlResults_28]:X %controlResults_22#3 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_32, %result_33 = qcirc.gate1q[%controlResults_30]:X %controlResults_22#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_34, %result_35 = qcirc.gate1q[%controlResults_32]:X %controlResults_22#5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_36, %result_37 = qcirc.gate1q[%controlResults_34]:X %controlResults_22#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_38, %result_39 = qcirc.gate1q[%controlResults_36]:X %controlResults_22#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_40, %result_41 = qcirc.gate1q[%controlResults_38]:X %controlResults_22#8 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_42, %result_43 = qcirc.gate1q[%controlResults_40]:X %controlResults_22#9 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_44, %result_45 = qcirc.gate1q[%controlResults_42]:X %controlResults_22#10 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_46:11, %result_47 = qcirc.gate1q[%result_25, %result_27, %result_29, %result_31, %result_33, %result_35, %result_37, %result_39, %result_41, %result_43, %result_45]:X %controlResults_44 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_48 = qcirc.gate1q[]:X %controlResults_46#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_49 = qcirc.gate1q[]:X %controlResults_46#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_50 = qcirc.gate1q[]:X %controlResults_46#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_51 = qcirc.gate1q[]:X %controlResults_46#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_52 = qcirc.gate1q[]:X %controlResults_46#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_53 = qcirc.gate1q[]:X %controlResults_46#5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_54 = qcirc.gate1q[]:X %controlResults_46#6 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_55 = qcirc.gate1q[]:X %controlResults_46#7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_56 = qcirc.gate1q[]:X %controlResults_46#8 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_57 = qcirc.gate1q[]:X %controlResults_46#9 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_58 = qcirc.gate1q[]:X %controlResults_46#10 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_59:11, %result_60 = qcirc.gate1q[%result_48, %result_49, %result_50, %result_51, %result_52, %result_53, %result_54, %result_55, %result_56, %result_57, %result_58]:X %result_47 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_61 = qcirc.gate1q[]:X %controlResults_59#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_62 = qcirc.gate1q[]:X %controlResults_59#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_63 = qcirc.gate1q[]:X %controlResults_59#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_64 = qcirc.gate1q[]:X %controlResults_59#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_65 = qcirc.gate1q[]:X %controlResults_59#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_66 = qcirc.gate1q[]:X %controlResults_59#5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_67 = qcirc.gate1q[]:X %controlResults_59#6 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_68 = qcirc.gate1q[]:X %controlResults_59#7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_69 = qcirc.gate1q[]:X %controlResults_59#8 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_70 = qcirc.gate1q[]:X %controlResults_59#9 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_71 = qcirc.gate1q[]:X %controlResults_59#10 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   qcirc.qfreez %result_60 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%result_61) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_72, %measResult_73 = qcirc.measure(%result_62) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_72 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_74, %measResult_75 = qcirc.measure(%result_63) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_74 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_76, %measResult_77 = qcirc.measure(%result_64) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_76 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_78, %measResult_79 = qcirc.measure(%result_65) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_78 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_80, %measResult_81 = qcirc.measure(%result_66) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_80 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_82, %measResult_83 = qcirc.measure(%result_67) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_82 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_84, %measResult_85 = qcirc.measure(%result_68) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_84 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_86, %measResult_87 = qcirc.measure(%result_69) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_86 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_88, %measResult_89 = qcirc.measure(%result_70) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_88 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_90, %measResult_91 = qcirc.measure(%result_71) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_90 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %12 = qcirc.arrpack(%measResult, %measResult_73, %measResult_75, %measResult_77, %measResult_79, %measResult_81, %measResult_83, %measResult_85, %measResult_87, %measResult_89, %measResult_91) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) -> !qcirc<array<i1>[11]>
//  CHECK-NEXT:   return %12 : !qcirc<array<i1>[11]>
//  CHECK-NEXT: }
qwerty.func @big_perm[]() irrev-> !qwerty<bitbundle[11]> {
  %0 = qwerty.qbprep Z<PLUS>[11] : () -> !qwerty<qbundle[11]>
  %1 = qwerty.qbtrans %0 by {list:{"|00000000000>", "|11111111111>"}} >> {list:{"|11111111111>", "|00000000000>"}} : (!qwerty<qbundle[11]>) -> !qwerty<qbundle[11]>
  %2 = qwerty.qbmeas %1 by {std: Z[11]} : !qwerty<qbundle[11]> -> !qwerty<bitbundle[11]>
  qwerty.return %2 : !qwerty<bitbundle[11]>
}

// The tweedledum ("slow") synthesis would crash the program. Let's not bother
// checking the output for correctness; instead, just make sure this doesn't
// crash.
// CHECK-LABEL: func.func @really_big_perm() -> !qcirc<array<i1>[128]> {
qwerty.func @really_big_perm[]() irrev-> !qwerty<bitbundle[128]> {
  %0 = qwerty.qbprep Z<PLUS>[128] : () -> !qwerty<qbundle[128]>
  %1 = qwerty.qbtrans %0 by {list:{"|00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000>", "|11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111>"}} >> {list:{"|11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111>", "|00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000>"}} : (!qwerty<qbundle[128]>) -> !qwerty<qbundle[128]>
  %2 = qwerty.qbmeas %1 by {std: Z[128]} : !qwerty<qbundle[128]> -> !qwerty<bitbundle[128]>
  qwerty.return %2 : !qwerty<bitbundle[128]>
}

// CHECK-LABEL: func.func @big_perm_multi_mask() -> !qcirc<array<i1>[11]> {
//  CHECK-NEXT:   %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %3 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %4 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %5 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %6 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %7 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %8 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %9 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %10 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %11 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %12 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result = qcirc.gate1q[]:X %2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults:11, %result_0 = qcirc.gate1q[%0, %1, %result, %3, %4, %5, %6, %7, %8, %9, %10]:X %11 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:X %controlResults#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:X %controlResults#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_3:11, %result_4 = qcirc.gate1q[%controlResults#0, %result_2, %result_1, %controlResults#3, %controlResults#4, %controlResults#5, %controlResults#6, %controlResults#7, %controlResults#8, %controlResults#9, %controlResults#10]:X %result_0 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:X %controlResults_3#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_6:11, %result_7 = qcirc.gate1q[%controlResults_3#0, %result_5, %controlResults_3#2, %controlResults_3#3, %controlResults_3#4, %controlResults_3#5, %controlResults_3#6, %controlResults_3#7, %controlResults_3#8, %controlResults_3#9, %controlResults_3#10]:X %12 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_8 = qcirc.gate1q[]:X %controlResults_6#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:X %controlResults_6#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_10 = qcirc.gate1q[]:X %controlResults_6#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_11:11, %result_12 = qcirc.gate1q[%result_8, %result_9, %result_10, %controlResults_6#3, %controlResults_6#4, %controlResults_6#5, %controlResults_6#6, %controlResults_6#7, %controlResults_6#8, %controlResults_6#9, %controlResults_6#10]:X %result_7 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_13 = qcirc.gate1q[]:X %controlResults_11#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_14 = qcirc.gate1q[]:X %controlResults_11#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_15 = qcirc.gate1q[]:X %controlResults_11#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_16, %result_17 = qcirc.gate1q[%result_4]:X %result_14 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_18, %result_19 = qcirc.gate1q[%controlResults_16]:X %result_15 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_20, %result_21 = qcirc.gate1q[%result_12]:X %result_13 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_22, %result_23 = qcirc.gate1q[%controlResults_20]:X %result_17 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_24, %result_25 = qcirc.gate1q[%controlResults_22]:X %result_19 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_26 = qcirc.gate1q[]:X %result_23 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_27:11, %result_28 = qcirc.gate1q[%result_21, %result_26, %result_25, %controlResults_11#3, %controlResults_11#4, %controlResults_11#5, %controlResults_11#6, %controlResults_11#7, %controlResults_11#8, %controlResults_11#9, %controlResults_11#10]:X %controlResults_18 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_29 = qcirc.gate1q[]:X %controlResults_27#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_30 = qcirc.gate1q[]:X %controlResults_27#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_31:11, %result_32 = qcirc.gate1q[%result_29, %controlResults_27#1, %result_30, %controlResults_27#3, %controlResults_27#4, %controlResults_27#5, %controlResults_27#6, %controlResults_27#7, %controlResults_27#8, %controlResults_27#9, %controlResults_27#10]:X %controlResults_24 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_33 = qcirc.gate1q[]:X %controlResults_31#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_34 = qcirc.gate1q[]:X %controlResults_31#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_35:11, %result_36 = qcirc.gate1q[%result_33, %result_34, %controlResults_31#2, %controlResults_31#3, %controlResults_31#4, %controlResults_31#5, %controlResults_31#6, %controlResults_31#7, %controlResults_31#8, %controlResults_31#9, %controlResults_31#10]:X %result_28 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_37 = qcirc.gate1q[]:X %controlResults_35#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_38:11, %result_39 = qcirc.gate1q[%controlResults_35#0, %controlResults_35#1, %result_37, %controlResults_35#3, %controlResults_35#4, %controlResults_35#5, %controlResults_35#6, %controlResults_35#7, %controlResults_35#8, %controlResults_35#9, %controlResults_35#10]:X %result_32 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_36 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_39 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%controlResults_38#0) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_40, %measResult_41 = qcirc.measure(%controlResults_38#1) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_40 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_42, %measResult_43 = qcirc.measure(%controlResults_38#2) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_42 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_44, %measResult_45 = qcirc.measure(%controlResults_38#3) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_44 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_46, %measResult_47 = qcirc.measure(%controlResults_38#4) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_46 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_48, %measResult_49 = qcirc.measure(%controlResults_38#5) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_48 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_50, %measResult_51 = qcirc.measure(%controlResults_38#6) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_50 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_52, %measResult_53 = qcirc.measure(%controlResults_38#7) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_52 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_54, %measResult_55 = qcirc.measure(%controlResults_38#8) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_54 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_56, %measResult_57 = qcirc.measure(%controlResults_38#9) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_56 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_58, %measResult_59 = qcirc.measure(%controlResults_38#10) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_58 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %13 = qcirc.arrpack(%measResult, %measResult_41, %measResult_43, %measResult_45, %measResult_47, %measResult_49, %measResult_51, %measResult_53, %measResult_55, %measResult_57, %measResult_59) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) -> !qcirc<array<i1>[11]>
//  CHECK-NEXT:   return %13 : !qcirc<array<i1>[11]>
//  CHECK-NEXT: }
qwerty.func @big_perm_multi_mask[]() irrev-> !qwerty<bitbundle[11]> {
  %0 = qwerty.qbprep Z<PLUS>[11] : () -> !qwerty<qbundle[11]>
  %1 = qwerty.qbtrans %0 by {list:{"|11011111111>", "|11111111111>", "|10111111111>", "|00011111111>"}} >> {list:{"|10111111111>", "|00011111111>", "|11011111111>", "|11111111111>"}} : (!qwerty<qbundle[11]>) -> !qwerty<qbundle[11]>
  %2 = qwerty.qbmeas %1 by {std: Z[11]} : !qwerty<qbundle[11]> -> !qwerty<bitbundle[11]>
  qwerty.return %2 : !qwerty<bitbundle[11]>
}

// CHECK-LABEL: func.func @revolve_3q_general() -> !qcirc<array<i1>[3]> {
//  CHECK-NEXT:  %0 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:  %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:  %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:  %result = qcirc.gate1q[]:Sdg %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:  %result_0 = qcirc.gate1q[]:Sdg %1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:  %result_1 = qcirc.gate1q[]:H %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:  %result_2 = qcirc.gate1q[]:Sdg %2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:  %result_3 = qcirc.gate1q[]:H %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:  %controlResults, %result_4 = qcirc.gate1q[%result_1]:S %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:  %controlResults_5, %result_6 = qcirc.gate1q[%result_3]:T %result_4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:  %result_7 = qcirc.gate1q[]:H %result_6 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:  %qubitResult, %measResult = qcirc.measure(%controlResults) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:  qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:  %qubitResult_8, %measResult_9 = qcirc.measure(%controlResults_5) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:  qcirc.qfree %qubitResult_8 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:  %qubitResult_10, %measResult_11 = qcirc.measure(%result_7) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:  qcirc.qfree %qubitResult_10 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:  %3 = qcirc.arrpack(%measResult, %measResult_9, %measResult_11) : (i1, i1, i1) -> !qcirc<array<i1>[3]>
//  CHECK-NEXT:  return %3 : !qcirc<array<i1>[3]>
//  CHECK-NEXT: }

qwerty.func @revolve_3q_general[]() irrev-> !qwerty<bitbundle[3]> {
  %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %2 = qwerty.qbunpack %0 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %3 = qwerty.qbunpack %1 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %4 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %5 = qwerty.qbunpack %4 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
  %6 = qwerty.qbpack(%2, %3, %5) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
  %7 = qwerty.qbtrans %6 by {std: Y[3]} >> {revolve: {std: Z[2]} by {"|p>", "|m>"}} : (!qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
  %8:3 = qwerty.qbunpack %7 : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
  %9 = qwerty.qbpack(%8#0) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
  %10 = qwerty.qbmeas %9 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %11 = qwerty.bitunpack %10 : (!qwerty<bitbundle[1]>) -> i1
  %12 = qwerty.qbpack(%8#1) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
  %13 = qwerty.qbmeas %12 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %14 = qwerty.bitunpack %13 : (!qwerty<bitbundle[1]>) -> i1
  %15 = qwerty.qbpack(%8#2) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
  %16 = qwerty.qbmeas %15 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %17 = qwerty.bitunpack %16 : (!qwerty<bitbundle[1]>) -> i1
  %18 = qwerty.bitpack(%11, %14, %17) : (i1, i1, i1) -> !qwerty<bitbundle[3]>
  qwerty.return %18 : !qwerty<bitbundle[3]>
}
