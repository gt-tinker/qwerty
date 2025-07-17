// RUN: qwerty-opt -split-input-file -inline %s | FileCheck %s

qwerty.func private @h[](%arg0: !qwerty<qbundle[2]>) rev -> !qwerty<qbundle[2]> {
  %0 = qwerty.qbtrans %arg0 by {std:Z[2]} >> {std:X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @main[]() irrev-> !qwerty<bitbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %1 = qwerty.qbtrans %0 by {std: X[2]} >> {std: Z[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %2 = qwerty.qbmeas %1 by {std: Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
//  CHECK-NEXT:   qwerty.return %2 : !qwerty<bitbundle[2]>
//  CHECK-NEXT: }
qwerty.func @main[]() irrev -> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
  %1 = qwerty.call adj @h (%0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbmeas %1 by {std:Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %2 : !qwerty<bitbundle[2]>
}

// -----

qwerty.func private @h[](%arg0: !qwerty<qbundle[2]>) rev -> !qwerty<qbundle[2]> {
  %0 = qwerty.qbtrans %arg0 by {std:Z[2]} >> {std:X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @main[]() irrev-> !qwerty<bitbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %1:5 = qwerty.qbunpack %0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %2 = qwerty.qbpack(%1#3, %1#4) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %3 = qwerty.qbtrans %2 by {std: Z[2]} >> {std: X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %4:2 = qwerty.qbunpack %3 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %5 = qwerty.qbpack(%1#0, %1#1, %1#2, %4#0, %4#1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %6 = qwerty.qbmeas %5 by {std: Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
//  CHECK-NEXT:   qwerty.return %6 : !qwerty<bitbundle[5]>
//  CHECK-NEXT: }
qwerty.func @main[]() irrev -> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
  %1 = qwerty.call pred {std:X[3]} @h (%0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %2 = qwerty.qbmeas %1 by {std:Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
  qwerty.return %2 : !qwerty<bitbundle[5]>
}

// -----

qwerty.func private @h[](%arg0: !qwerty<qbundle[2]>) rev -> !qwerty<qbundle[2]> {
  %0 = qwerty.qbtrans %arg0 by {std:Z[2]} >> {std:X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @main[]() irrev-> !qwerty<bitbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %1:5 = qwerty.qbunpack %0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %2 = qwerty.qbpack(%1#0, %1#1, %1#2, %1#3, %1#4) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %3 = qwerty.qbtrans %2 by {list:{"|000>"}, std: Z[2]} >> {list:{"|000>"}, std: X[2]} : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %4:5 = qwerty.qbunpack %3 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %5 = qwerty.qbpack(%4#0, %4#1, %4#2, %4#3, %4#4) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %6 = qwerty.qbmeas %5 by {std: Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
//  CHECK-NEXT:   qwerty.return %6 : !qwerty<bitbundle[5]>
//  CHECK-NEXT: }
qwerty.func @main[]() irrev -> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
  %1 = qwerty.call pred {list:{"|000>"}} @h (%0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %2 = qwerty.qbmeas %1 by {std:Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
  qwerty.return %2 : !qwerty<bitbundle[5]>
}

// -----

qwerty.func private @h[](%arg0: !qwerty<qbundle[2]>) rev -> !qwerty<qbundle[2]> {
  %0:2 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
  %1 = qwerty.qbpack(%0#1, %0#0) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbtrans %1 by {std:Z[2]} >> {std:X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %2 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @main[]() irrev-> !qwerty<bitbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %1:5 = qwerty.qbunpack %0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %2 = qwerty.qbpack(%1#0, %1#1, %1#2, %1#4, %1#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %3 = qwerty.qbtrans %2 by {list:{"|111>"}, std: Z[2]} >> {list:{"|111>"}, std: X[2]} : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %4:5 = qwerty.qbunpack %3 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %leftResult, %rightResult = qcirc.gate2q[]:Swap %4#3, %4#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults:3, %leftResult_0, %rightResult_1 = qcirc.gate2q[%4#0, %4#1, %4#2]:Swap %leftResult, %rightResult : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %5 = qwerty.qbpack(%controlResults#0, %controlResults#1, %controlResults#2, %leftResult_0, %rightResult_1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %6 = qwerty.qbmeas %5 by {std: Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
//  CHECK-NEXT:   qwerty.return %6 : !qwerty<bitbundle[5]>
//  CHECK-NEXT: }
qwerty.func @main[]() irrev -> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
  %1 = qwerty.call pred {list:{"|111>"}} @h (%0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %2 = qwerty.qbmeas %1 by {std:Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
  qwerty.return %2 : !qwerty<bitbundle[5]>
}

// -----

qwerty.func private @h[](%arg0: !qwerty<qbundle[2]>) rev -> !qwerty<qbundle[2]> {
  %0:2 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
  %1:2 = qcirc.gate2q[]:Swap %0#0, %0#1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
  %2 = qwerty.qbpack(%1#0, %1#1) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
  qwerty.return %2 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @main[]() irrev-> !qwerty<bitbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %1:5 = qwerty.qbunpack %0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result = qcirc.gate1q[]:X %1#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:X %1#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:X %1#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults:3, %leftResult, %rightResult = qcirc.gate2q[%result, %result_0, %result_1]:Swap %1#3, %1#4 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:X %controlResults#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:X %controlResults#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_4 = qcirc.gate1q[]:X %controlResults#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qwerty.qbpack(%result_2, %result_3, %result_4, %leftResult, %rightResult) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %3 = qwerty.qbmeas %2 by {std: Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
//  CHECK-NEXT:   qwerty.return %3 : !qwerty<bitbundle[5]>
//  CHECK-NEXT: }
qwerty.func @main[]() irrev -> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbprep Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
  %1 = qwerty.call pred {list:{"|000>"}} @h (%0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %2 = qwerty.qbmeas %1 by {std:Z[5]} : !qwerty<qbundle[5]> -> !qwerty<bitbundle[5]>
  qwerty.return %2 : !qwerty<bitbundle[5]>
}
