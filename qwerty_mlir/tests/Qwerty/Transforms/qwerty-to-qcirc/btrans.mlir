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
//  CHECK-NEXT:   %leftResult, %rightResult = qcirc.gate2q[]:Swap %1, %2 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result = qcirc.gate1q[]:H %rightResult : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults, %result_0 = qcirc.gate1q[%result]:Sdg %leftResult : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:H %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:H %0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_3, %result_4 = qcirc.gate1q[%result_1]:S %result_2 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:H %controlResults_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %leftResult_6, %rightResult_7 = qcirc.gate2q[]:Swap %result_4, %result_5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %qubitResult, %measResult = qcirc.measure(%leftResult_6) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_8, %measResult_9 = qcirc.measure(%rightResult_7) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_8 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %qubitResult_10, %measResult_11 = qcirc.measure(%controlResults) : (!qcirc.qubit) -> (!qcirc.qubit, i1)
//  CHECK-NEXT:   qcirc.qfree %qubitResult_10 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %3 = qcirc.arrpack(%measResult, %measResult_9, %measResult_11) : (i1, i1, i1) -> !qcirc<array<i1>[3]>
//  CHECK-NEXT:   return %3 : !qcirc<array<i1>[3]>
//  CHECK-NEXT: }
qwerty.func @qft[]() irrev-> !qwerty<bitbundle[3]> {
  %0 = qwerty.qbprep Z<PLUS>[3] : () -> !qwerty<qbundle[3]>
  %1 = qwerty.qbtrans %0 by {std:Z[1], std:FOURIER[2]} >> {std:FOURIER[2], std:Z[1]} : (!qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
  %2 = qwerty.qbmeas %1 by {std:Z[3]} : !qwerty<qbundle[3]> -> !qwerty<bitbundle[3]>
  qwerty.return %2 : !qwerty<bitbundle[3]>
}

// CHECK-LABEL: func.func @fourier_identity() -> !qcirc<array<i1>[3]> {
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
qwerty.func @fourier_identity[]() irrev-> !qwerty<bitbundle[3]> {
  %0 = qwerty.qbprep Z<PLUS>[3] : () -> !qwerty<qbundle[3]>
  %1 = qwerty.qbtrans %0 by {std:FOURIER[3]} >> {std:FOURIER[3]} : (!qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
  %2 = qwerty.qbmeas %1 by {std:Z[3]} : !qwerty<qbundle[3]> -> !qwerty<bitbundle[3]>
  qwerty.return %2 : !qwerty<bitbundle[3]>
}

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
