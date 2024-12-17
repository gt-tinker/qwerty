// RUN: qwerty-opt -canonicalize %s | FileCheck %s

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

qwerty.func private @id[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.qbid %arg0 : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @call_const_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.call adj @trivial(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @call_const_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.call_indirect %1(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %2 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @call_const_adj_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.call @trivial(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @call_const_adj_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.func_adj %1 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %3 = qwerty.call_indirect %2(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %3 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @return_adj_adj[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
//  CHECK-NEXT:   qwerty.return %arg0 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT: }
qwerty.func @return_adj_adj[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
  %0 = qwerty.func_adj %arg0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  qwerty.return %1 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
}

// CHECK-LABEL: qwerty.func @return_pred_pred[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>> {
//  CHECK-NEXT:   %0 = qwerty.func_pred %arg0 by {list:{"|j>"}, list:{"|1>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT: }
qwerty.func @return_pred_pred[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|1>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>
  %1 = qwerty.func_pred %0 by {list:{"|j>"}} : (!qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  qwerty.return %1 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
}

// CHECK-LABEL: qwerty.func @return_pointless_pred[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>> {
//  CHECK-NEXT:   %0 = qwerty.lambda[%arg0 as %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>](%arg2: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:     %1:4 = qwerty.qbunpack %arg2 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %2 = qwerty.qbpack(%1#2, %1#3) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     %3 = qwerty.call_indirect %arg1(%2) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     %4:2 = qwerty.qbunpack %3 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %5 = qwerty.qbpack(%1#0, %1#1, %4#0, %4#1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:     qwerty.return %5 : !qwerty<qbundle[4]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT: }
qwerty.func @return_pointless_pred[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>> {
  %0 = qwerty.func_pred %arg0 by {std:X[2]} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
}

// CHECK-LABEL: qwerty.func @return_semi_pointless_pred[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>> {
//  CHECK-NEXT:   %0 = qwerty.lambda[%arg0 as %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>](%arg2: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:     %1:4 = qwerty.qbunpack %arg2 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %2 = qwerty.qbpack(%1#0, %1#2, %1#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
//  CHECK-NEXT:     %3 = qwerty.func_pred %arg1 by {list:{"|j>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>
//  CHECK-NEXT:     %4 = qwerty.call_indirect %3(%2) : (!qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>, !qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
//  CHECK-NEXT:     %5:3 = qwerty.qbunpack %4 : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %6 = qwerty.qbpack(%5#0, %1#1, %5#1, %5#2) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:     qwerty.return %6 : !qwerty<qbundle[4]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT: }
qwerty.func @return_semi_pointless_pred[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|j>"}, std:X[1]} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
}

// CHECK-LABEL: qwerty.func @call_pointless_pred[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0:4 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qwerty.qbpack(%0#2, %0#3) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %2 = qwerty.call @trivial(%1) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %3:2 = qwerty.qbunpack %2 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qwerty.qbpack(%0#0, %0#1, %3#0, %3#1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.return %4 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
qwerty.func @call_pointless_pred[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  %0 = qwerty.call pred {std:X[2]} @trivial(%arg0) : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %0 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: qwerty.func @call_semi_pointless_pred[](%arg0: !qwerty<qbundle[6]>) rev-> !qwerty<qbundle[6]> {
//  CHECK-NEXT:   %0:6 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[6]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qwerty.qbpack(%0#0, %0#1, %0#4, %0#5) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %2 = qwerty.call pred {list:{"|pm>"}} @trivial(%1) : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %3:4 = qwerty.qbunpack %2 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qwerty.qbpack(%3#0, %3#1, %0#2, %0#3, %3#2, %3#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[6]>
//  CHECK-NEXT:   qwerty.return %4 : !qwerty<qbundle[6]>
//  CHECK-NEXT: }
qwerty.func @call_semi_pointless_pred[](%arg0: !qwerty<qbundle[6]>) rev-> !qwerty<qbundle[6]> {
  %0 = qwerty.call pred {list:{"|pm>"}, std:X[2]} @trivial(%arg0) : (!qwerty<qbundle[6]>) -> !qwerty<qbundle[6]>
  qwerty.return %0 : !qwerty<qbundle[6]>
}

// CHECK-LABEL: qwerty.func @call_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = scf.if %arg0 -> (!qwerty<qbundle[2]>) {
//  CHECK-NEXT:     %1 = qwerty.call @trivial(%arg1) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     scf.yield %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %1 = qwerty.call @id(%arg1) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     scf.yield %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @call_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
  %0 = scf.if %arg0 -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
    %2 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } else {
    %3 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %3 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  }
  %1 = qwerty.call_indirect %0(%arg1) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @adj_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
//  CHECK-NEXT:   %0 = scf.if %arg0 -> (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) {
//  CHECK-NEXT:     %1 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:     %2 = qwerty.func_adj %1 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:     scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %1 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:     %2 = qwerty.func_adj %1 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:     scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT: }
qwerty.func @adj_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
  %0 = scf.if %arg0 -> (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) {
    %2 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } else {
    %2 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  }
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  qwerty.return %1 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
}

// CHECK-LABEL: qwerty.func @adj_call_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = scf.if %arg0 -> (!qwerty<qbundle[2]>) {
//  CHECK-NEXT:     %1 = qwerty.call adj @trivial(%arg1) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     scf.yield %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %1 = qwerty.call adj @id(%arg1) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     scf.yield %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @adj_call_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
  %0 = scf.if %arg0 -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
    %2 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } else {
    %3 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %3 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  }
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.call_indirect %1(%arg1) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %2 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @pred_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
//  CHECK-NEXT:   %0 = scf.if %arg0 -> (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>) {
//  CHECK-NEXT:     %1 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:     %2 = qwerty.func_pred %1 by {list:{"|ppmpp>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT:     scf.yield %2 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %1 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:     %2 = qwerty.func_pred %1 by {list:{"|ppmpp>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT:     scf.yield %2 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT: }
qwerty.func @pred_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[2]>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
  %0 = scf.if %arg0 -> (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) {
    %2 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } else {
    %2 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  }
  %1 = qwerty.func_pred %0 by {list:{"|ppmpp>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
  qwerty.return %1 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
}

// CHECK-LABEL: qwerty.func @pred_adj_call_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0 = scf.if %arg0 -> (!qwerty<qbundle[5]>) {
//  CHECK-NEXT:     %1 = qwerty.call adj pred {list:{"|jjj>"}} @trivial(%arg1) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:     scf.yield %1 : !qwerty<qbundle[5]>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %1 = qwerty.call adj pred {list:{"|jjj>"}} @id(%arg1) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:     scf.yield %1 : !qwerty<qbundle[5]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
qwerty.func @pred_adj_call_if_const[](%arg0: i1, %arg1: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
  %0 = scf.if %arg0 -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>> {
    %2 = qwerty.func_const @trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } else {
    %3 = qwerty.func_const @id[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield %3 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  }
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.func_pred %1 by {list:{"|jjj>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %3 = qwerty.call_indirect %2(%arg1) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %3 : !qwerty<qbundle[5]>
}
