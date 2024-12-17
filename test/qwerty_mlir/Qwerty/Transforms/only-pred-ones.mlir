// RUN: qwerty-opt -only-pred-ones %s | FileCheck %s

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @func_pred_pointless[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
//  CHECK-NEXT:   %0 = qwerty.lambda[%arg0 as %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>](%arg2: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
//  CHECK-NEXT:     %1:7 = qwerty.qbunpack %arg2 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %2 = qwerty.qbpack(%1#5, %1#6) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     %3 = qwerty.call_indirect %arg1(%2) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:     %4:2 = qwerty.qbunpack %3 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %5 = qwerty.qbpack(%1#0, %1#1, %1#2, %1#3, %1#4, %4#0, %4#1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     qwerty.return %5 : !qwerty<qbundle[7]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT: }
qwerty.func @func_pred_pointless[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|ji>","|ij>","|ii>","|jj>"}, std:Y[3]} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
}

// CHECK-LABEL: qwerty.func @call_pointless[](%arg0: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
//  CHECK-NEXT:   %0:7 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qwerty.qbpack(%0#5, %0#6) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %2 = qwerty.call @trivial(%1) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   %3:2 = qwerty.qbunpack %2 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qwerty.qbpack(%0#0, %0#1, %0#2, %0#3, %0#4, %3#0, %3#1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   qwerty.return %4 : !qwerty<qbundle[7]>
//  CHECK-NEXT: }
qwerty.func @call_pointless[](%arg0: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
  %0 = qwerty.call pred {list:{"|ji>","|ij>","|ii>","|jj>"}, std:Y[3]} @trivial(%arg0) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
  qwerty.return %0 : !qwerty<qbundle[7]>
}

// CHECK-LABEL: qwerty.func @func_pred_semi_pointless[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
//  CHECK-NEXT:   %0 = qwerty.lambda[%arg0 as %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>](%arg2: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
//  CHECK-NEXT:     %1:7 = qwerty.qbunpack %arg2 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result = qcirc.gate1q[]:Sdg %1#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_1 = qcirc.gate1q[]:X %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_2 = qcirc.gate1q[]:Sdg %1#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_3 = qcirc.gate1q[]:H %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %2 = qwerty.func_pred %arg1 by {list:{"|11>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT:     %3 = qwerty.qbpack(%result_1, %result_3, %1#5, %1#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:     %4 = qwerty.call_indirect %2(%3) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:     %5:4 = qwerty.qbunpack %4 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_4 = qcirc.gate1q[]:X %5#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_5 = qcirc.gate1q[]:H %result_4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_6 = qcirc.gate1q[]:S %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_7 = qcirc.gate1q[]:H %5#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_8 = qcirc.gate1q[]:S %result_7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %6 = qwerty.qbpack(%result_6, %result_8, %1#2, %1#3, %1#4, %5#2, %5#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     qwerty.return %6 : !qwerty<qbundle[7]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT: }
qwerty.func @func_pred_semi_pointless[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|ij>"}, std:X[3]} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
}

// CHECK-LABEL: qwerty.func @call_semi_pointless[](%arg0: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
//  CHECK-NEXT:   %0:7 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sdg %0#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:X %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:Sdg %0#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:H %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%result_1, %result_3, %0#5, %0#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %2 = qwerty.call pred {list:{"|11>"}} @trivial(%1) : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %3:4 = qwerty.qbunpack %2 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_4 = qcirc.gate1q[]:X %3#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:H %result_4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:S %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_7 = qcirc.gate1q[]:H %3#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_8 = qcirc.gate1q[]:S %result_7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %4 = qwerty.qbpack(%result_6, %result_8, %0#2, %0#3, %0#4, %3#2, %3#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   qwerty.return %4 : !qwerty<qbundle[7]>
//  CHECK-NEXT: }
qwerty.func @call_semi_pointless[](%arg0: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
  %0 = qwerty.call pred {list:{"|ij>"}, std:X[3]} @trivial(%arg0) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
  qwerty.return %0 : !qwerty<qbundle[7]>
}

// CHECK-LABEL: qwerty.func @func_pred_already_ones[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>> {
//  CHECK-NEXT:   %0 = qwerty.func_pred %arg0 by {list:{"|11>"}, list:{"|1>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
//  CHECK-NEXT: }
qwerty.func @func_pred_already_ones[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|11>"}, list:{"|1>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
}

// CHECK-LABEL: qwerty.func @call_already_ones[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0 = qwerty.call pred {list:{"|11>"}, list:{"|1>"}} @trivial(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
qwerty.func @call_already_ones[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.call pred {list:{"|11>"}, list:{"|1>"}} @trivial(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %0 : !qwerty<qbundle[5]>
}

// CHECK-LABEL: qwerty.func @func_pred_cartesian_product[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
//  CHECK-NEXT:   %0 = qwerty.lambda[%arg0 as %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>](%arg2: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
//  CHECK-NEXT:     %1:7 = qwerty.qbunpack %arg2 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result = qcirc.gate1q[]:Sdg %1#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_1 = qcirc.gate1q[]:X %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_2 = qcirc.gate1q[]:Sdg %1#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_3 = qcirc.gate1q[]:H %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_4 = qcirc.gate1q[]:X %result_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_5 = qcirc.gate1q[]:H %1#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_6 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_7 = qcirc.gate1q[]:H %1#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_8 = qcirc.gate1q[]:X %result_7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_9 = qcirc.gate1q[]:H %1#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %2 = qwerty.func_pred %arg1 by {list:{"|11111>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT:     %3 = qwerty.qbpack(%result_1, %result_4, %result_6, %result_8, %result_9, %1#5, %1#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %4 = qwerty.call_indirect %2(%3) : (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>, !qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %5:7 = qwerty.qbunpack %4 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_10 = qcirc.gate1q[]:X %5#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_11 = qcirc.gate1q[]:H %result_10 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_12 = qcirc.gate1q[]:S %result_11 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_13 = qcirc.gate1q[]:X %5#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_14 = qcirc.gate1q[]:H %result_13 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_15 = qcirc.gate1q[]:S %result_14 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_16 = qcirc.gate1q[]:X %5#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_17 = qcirc.gate1q[]:H %result_16 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_18 = qcirc.gate1q[]:X %5#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_19 = qcirc.gate1q[]:H %result_18 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_20 = qcirc.gate1q[]:H %5#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_21 = qcirc.gate1q[]:Sdg %result_12 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_22 = qcirc.gate1q[]:H %result_21 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_23 = qcirc.gate1q[]:Sdg %result_15 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_24 = qcirc.gate1q[]:H %result_23 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_25 = qcirc.gate1q[]:X %result_24 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_26 = qcirc.gate1q[]:H %result_17 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_27 = qcirc.gate1q[]:X %result_26 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_28 = qcirc.gate1q[]:H %result_19 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_29 = qcirc.gate1q[]:X %result_28 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_30 = qcirc.gate1q[]:H %result_20 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %6 = qwerty.qbpack(%result_22, %result_25, %result_27, %result_29, %result_30, %5#5, %5#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %7 = qwerty.call_indirect %2(%6) : (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>, !qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %8:7 = qwerty.qbunpack %7 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_31 = qcirc.gate1q[]:H %8#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_32 = qcirc.gate1q[]:S %result_31 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_33 = qcirc.gate1q[]:X %8#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_34 = qcirc.gate1q[]:H %result_33 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_35 = qcirc.gate1q[]:S %result_34 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_36 = qcirc.gate1q[]:X %8#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_37 = qcirc.gate1q[]:H %result_36 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_38 = qcirc.gate1q[]:X %8#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_39 = qcirc.gate1q[]:H %result_38 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_40 = qcirc.gate1q[]:H %8#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_41 = qcirc.gate1q[]:Sdg %result_32 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_42 = qcirc.gate1q[]:H %result_41 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_43 = qcirc.gate1q[]:X %result_42 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_44 = qcirc.gate1q[]:Sdg %result_35 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_45 = qcirc.gate1q[]:H %result_44 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_46 = qcirc.gate1q[]:X %result_45 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_47 = qcirc.gate1q[]:H %result_37 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_48 = qcirc.gate1q[]:H %result_39 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_49 = qcirc.gate1q[]:H %result_40 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %9 = qwerty.qbpack(%result_43, %result_46, %result_47, %result_48, %result_49, %8#5, %8#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %10 = qwerty.call_indirect %2(%9) : (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>, !qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %11:7 = qwerty.qbunpack %10 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_50 = qcirc.gate1q[]:X %11#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_51 = qcirc.gate1q[]:H %result_50 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_52 = qcirc.gate1q[]:S %result_51 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_53 = qcirc.gate1q[]:X %11#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_54 = qcirc.gate1q[]:H %result_53 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_55 = qcirc.gate1q[]:S %result_54 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_56 = qcirc.gate1q[]:H %11#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_57 = qcirc.gate1q[]:H %11#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_58 = qcirc.gate1q[]:H %11#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_59 = qcirc.gate1q[]:Sdg %result_52 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_60 = qcirc.gate1q[]:H %result_59 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_61 = qcirc.gate1q[]:Sdg %result_55 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_62 = qcirc.gate1q[]:H %result_61 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_63 = qcirc.gate1q[]:X %result_62 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_64 = qcirc.gate1q[]:H %result_56 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_65 = qcirc.gate1q[]:H %result_57 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_66 = qcirc.gate1q[]:H %result_58 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %12 = qwerty.qbpack(%result_60, %result_63, %result_64, %result_65, %result_66, %11#5, %11#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %13 = qwerty.call_indirect %2(%12) : (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>, !qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %14:7 = qwerty.qbunpack %13 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_67 = qcirc.gate1q[]:H %14#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_68 = qcirc.gate1q[]:S %result_67 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_69 = qcirc.gate1q[]:X %14#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_70 = qcirc.gate1q[]:H %result_69 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_71 = qcirc.gate1q[]:S %result_70 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_72 = qcirc.gate1q[]:H %14#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_73 = qcirc.gate1q[]:H %14#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_74 = qcirc.gate1q[]:H %14#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_75 = qcirc.gate1q[]:Sdg %result_68 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_76 = qcirc.gate1q[]:H %result_75 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_77 = qcirc.gate1q[]:X %result_76 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_78 = qcirc.gate1q[]:Sdg %result_71 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_79 = qcirc.gate1q[]:H %result_78 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_80 = qcirc.gate1q[]:X %result_79 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_81 = qcirc.gate1q[]:H %result_72 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_82 = qcirc.gate1q[]:H %result_73 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_83 = qcirc.gate1q[]:H %result_74 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_84 = qcirc.gate1q[]:X %result_83 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %15 = qwerty.qbpack(%result_77, %result_80, %result_81, %result_82, %result_84, %14#5, %14#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %16 = qwerty.call_indirect %2(%15) : (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>, !qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %17:7 = qwerty.qbunpack %16 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_85 = qcirc.gate1q[]:X %17#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_86 = qcirc.gate1q[]:H %result_85 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_87 = qcirc.gate1q[]:S %result_86 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_88 = qcirc.gate1q[]:X %17#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_89 = qcirc.gate1q[]:H %result_88 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_90 = qcirc.gate1q[]:S %result_89 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_91 = qcirc.gate1q[]:H %17#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_92 = qcirc.gate1q[]:H %17#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_93 = qcirc.gate1q[]:X %17#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_94 = qcirc.gate1q[]:H %result_93 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_95 = qcirc.gate1q[]:Sdg %result_87 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_96 = qcirc.gate1q[]:H %result_95 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_97 = qcirc.gate1q[]:Sdg %result_90 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_98 = qcirc.gate1q[]:H %result_97 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_99 = qcirc.gate1q[]:X %result_98 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_100 = qcirc.gate1q[]:H %result_91 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_101 = qcirc.gate1q[]:H %result_92 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_102 = qcirc.gate1q[]:H %result_94 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_103 = qcirc.gate1q[]:X %result_102 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %18 = qwerty.qbpack(%result_96, %result_99, %result_100, %result_101, %result_103, %17#5, %17#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %19 = qwerty.call_indirect %2(%18) : (!qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>, !qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     %20:7 = qwerty.qbunpack %19 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:     %result_104 = qcirc.gate1q[]:H %20#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_105 = qcirc.gate1q[]:S %result_104 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_106 = qcirc.gate1q[]:X %20#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_107 = qcirc.gate1q[]:H %result_106 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_108 = qcirc.gate1q[]:S %result_107 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_109 = qcirc.gate1q[]:H %20#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_110 = qcirc.gate1q[]:H %20#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_111 = qcirc.gate1q[]:X %20#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %result_112 = qcirc.gate1q[]:H %result_111 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:     %21 = qwerty.qbpack(%result_105, %result_108, %result_109, %result_110, %result_112, %20#5, %20#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:     qwerty.return %21 : !qwerty<qbundle[7]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
//  CHECK-NEXT: }
qwerty.func @func_pred_cartesian_product[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|ii>","|ji>"}, list:{"|ppm>","|mmm>","|mmp>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]>>
}

// CHECK-LABEL: qwerty.func @call_cartesian_product[](%arg0: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
//  CHECK-NEXT:   %0:7 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result = qcirc.gate1q[]:Sdg %0#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:H %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_1 = qcirc.gate1q[]:X %result_0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_2 = qcirc.gate1q[]:Sdg %0#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:H %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_4 = qcirc.gate1q[]:X %result_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_5 = qcirc.gate1q[]:H %0#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_7 = qcirc.gate1q[]:H %0#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_8 = qcirc.gate1q[]:X %result_7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:H %0#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%result_1, %result_4, %result_6, %result_8, %result_9, %0#5, %0#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %2 = qwerty.call pred {list:{"|11111>"}} @trivial(%1) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %3:7 = qwerty.qbunpack %2 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_10 = qcirc.gate1q[]:X %3#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_11 = qcirc.gate1q[]:H %result_10 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_12 = qcirc.gate1q[]:S %result_11 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_13 = qcirc.gate1q[]:X %3#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_14 = qcirc.gate1q[]:H %result_13 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_15 = qcirc.gate1q[]:S %result_14 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_16 = qcirc.gate1q[]:X %3#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_17 = qcirc.gate1q[]:H %result_16 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_18 = qcirc.gate1q[]:X %3#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_19 = qcirc.gate1q[]:H %result_18 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_20 = qcirc.gate1q[]:H %3#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_21 = qcirc.gate1q[]:Sdg %result_12 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_22 = qcirc.gate1q[]:H %result_21 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_23 = qcirc.gate1q[]:Sdg %result_15 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_24 = qcirc.gate1q[]:H %result_23 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_25 = qcirc.gate1q[]:X %result_24 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_26 = qcirc.gate1q[]:H %result_17 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_27 = qcirc.gate1q[]:X %result_26 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_28 = qcirc.gate1q[]:H %result_19 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_29 = qcirc.gate1q[]:X %result_28 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_30 = qcirc.gate1q[]:H %result_20 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %4 = qwerty.qbpack(%result_22, %result_25, %result_27, %result_29, %result_30, %3#5, %3#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %5 = qwerty.call pred {list:{"|11111>"}} @trivial(%4) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %6:7 = qwerty.qbunpack %5 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_31 = qcirc.gate1q[]:H %6#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_32 = qcirc.gate1q[]:S %result_31 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_33 = qcirc.gate1q[]:X %6#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_34 = qcirc.gate1q[]:H %result_33 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_35 = qcirc.gate1q[]:S %result_34 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_36 = qcirc.gate1q[]:X %6#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_37 = qcirc.gate1q[]:H %result_36 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_38 = qcirc.gate1q[]:X %6#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_39 = qcirc.gate1q[]:H %result_38 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_40 = qcirc.gate1q[]:H %6#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_41 = qcirc.gate1q[]:Sdg %result_32 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_42 = qcirc.gate1q[]:H %result_41 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_43 = qcirc.gate1q[]:X %result_42 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_44 = qcirc.gate1q[]:Sdg %result_35 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_45 = qcirc.gate1q[]:H %result_44 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_46 = qcirc.gate1q[]:X %result_45 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_47 = qcirc.gate1q[]:H %result_37 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_48 = qcirc.gate1q[]:H %result_39 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_49 = qcirc.gate1q[]:H %result_40 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %7 = qwerty.qbpack(%result_43, %result_46, %result_47, %result_48, %result_49, %6#5, %6#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %8 = qwerty.call pred {list:{"|11111>"}} @trivial(%7) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %9:7 = qwerty.qbunpack %8 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_50 = qcirc.gate1q[]:X %9#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_51 = qcirc.gate1q[]:H %result_50 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_52 = qcirc.gate1q[]:S %result_51 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_53 = qcirc.gate1q[]:X %9#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_54 = qcirc.gate1q[]:H %result_53 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_55 = qcirc.gate1q[]:S %result_54 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_56 = qcirc.gate1q[]:H %9#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_57 = qcirc.gate1q[]:H %9#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_58 = qcirc.gate1q[]:H %9#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_59 = qcirc.gate1q[]:Sdg %result_52 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_60 = qcirc.gate1q[]:H %result_59 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_61 = qcirc.gate1q[]:Sdg %result_55 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_62 = qcirc.gate1q[]:H %result_61 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_63 = qcirc.gate1q[]:X %result_62 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_64 = qcirc.gate1q[]:H %result_56 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_65 = qcirc.gate1q[]:H %result_57 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_66 = qcirc.gate1q[]:H %result_58 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %10 = qwerty.qbpack(%result_60, %result_63, %result_64, %result_65, %result_66, %9#5, %9#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %11 = qwerty.call pred {list:{"|11111>"}} @trivial(%10) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %12:7 = qwerty.qbunpack %11 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_67 = qcirc.gate1q[]:H %12#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_68 = qcirc.gate1q[]:S %result_67 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_69 = qcirc.gate1q[]:X %12#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_70 = qcirc.gate1q[]:H %result_69 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_71 = qcirc.gate1q[]:S %result_70 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_72 = qcirc.gate1q[]:H %12#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_73 = qcirc.gate1q[]:H %12#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_74 = qcirc.gate1q[]:H %12#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_75 = qcirc.gate1q[]:Sdg %result_68 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_76 = qcirc.gate1q[]:H %result_75 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_77 = qcirc.gate1q[]:X %result_76 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_78 = qcirc.gate1q[]:Sdg %result_71 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_79 = qcirc.gate1q[]:H %result_78 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_80 = qcirc.gate1q[]:X %result_79 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_81 = qcirc.gate1q[]:H %result_72 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_82 = qcirc.gate1q[]:H %result_73 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_83 = qcirc.gate1q[]:H %result_74 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_84 = qcirc.gate1q[]:X %result_83 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %13 = qwerty.qbpack(%result_77, %result_80, %result_81, %result_82, %result_84, %12#5, %12#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %14 = qwerty.call pred {list:{"|11111>"}} @trivial(%13) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %15:7 = qwerty.qbunpack %14 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_85 = qcirc.gate1q[]:X %15#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_86 = qcirc.gate1q[]:H %result_85 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_87 = qcirc.gate1q[]:S %result_86 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_88 = qcirc.gate1q[]:X %15#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_89 = qcirc.gate1q[]:H %result_88 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_90 = qcirc.gate1q[]:S %result_89 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_91 = qcirc.gate1q[]:H %15#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_92 = qcirc.gate1q[]:H %15#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_93 = qcirc.gate1q[]:X %15#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_94 = qcirc.gate1q[]:H %result_93 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_95 = qcirc.gate1q[]:Sdg %result_87 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_96 = qcirc.gate1q[]:H %result_95 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_97 = qcirc.gate1q[]:Sdg %result_90 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_98 = qcirc.gate1q[]:H %result_97 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_99 = qcirc.gate1q[]:X %result_98 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_100 = qcirc.gate1q[]:H %result_91 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_101 = qcirc.gate1q[]:H %result_92 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_102 = qcirc.gate1q[]:H %result_94 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_103 = qcirc.gate1q[]:X %result_102 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %16 = qwerty.qbpack(%result_96, %result_99, %result_100, %result_101, %result_103, %15#5, %15#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %17 = qwerty.call pred {list:{"|11111>"}} @trivial(%16) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   %18:7 = qwerty.qbunpack %17 : (!qwerty<qbundle[7]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_104 = qcirc.gate1q[]:H %18#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_105 = qcirc.gate1q[]:S %result_104 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_106 = qcirc.gate1q[]:X %18#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_107 = qcirc.gate1q[]:H %result_106 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_108 = qcirc.gate1q[]:S %result_107 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_109 = qcirc.gate1q[]:H %18#2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_110 = qcirc.gate1q[]:H %18#3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_111 = qcirc.gate1q[]:X %18#4 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_112 = qcirc.gate1q[]:H %result_111 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %19 = qwerty.qbpack(%result_105, %result_108, %result_109, %result_110, %result_112, %18#5, %18#6) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[7]>
//  CHECK-NEXT:   qwerty.return %19 : !qwerty<qbundle[7]>
//  CHECK-NEXT: }
qwerty.func @call_cartesian_product[](%arg0: !qwerty<qbundle[7]>) rev-> !qwerty<qbundle[7]> {
  %0 = qwerty.call pred {list:{"|ii>","|ji>"}, list:{"|ppm>","|mmm>","|mmp>"}} @trivial(%arg0) : (!qwerty<qbundle[7]>) -> !qwerty<qbundle[7]>
  qwerty.return %0 : !qwerty<qbundle[7]>
}
