// RUN: qwerty-opt -split-input-file -synth-embeds %s | FileCheck %s

// CHECK-LABEL: ccirc.circuit private @flip(%arg0: !ccirc<wirebundle[1]>) rev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wirebundle[1]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @flip__xor[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0:2 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#0]:X %0#1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:X %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults, %result_0) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @embed_xor1[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @flip__xor[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @embed_xor2[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @flip__xor[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
ccirc.circuit private @flip(%arg0: !ccirc<wirebundle[1]>) rev {
  %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
  ccirc.return %0 : !ccirc<wirebundle[1]>
}

qwerty.func @embed_xor1[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
  %0 = qwerty.embed_xor @flip : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

qwerty.func @embed_xor2[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
  %0 = qwerty.embed_xor @flip : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// -----

// CHECK-LABEL: ccirc.circuit private @bv_oracle(%arg0: !ccirc<wirebundle[4]>) irrev {
//  CHECK-NEXT:   %0 = ccirc.constant -3 : i4 : !ccirc<wirebundle[4]>
//  CHECK-NEXT:   %1 = ccirc.and(%arg0, %0) : (!ccirc<wirebundle[4]>, !ccirc<wirebundle[4]>) -> !ccirc<wirebundle[4]>
//  CHECK-NEXT:   %2:4 = ccirc.wireunpack %1 : (!ccirc<wirebundle[4]>) -> (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>)
//  CHECK-NEXT:   %3 = ccirc.xor(%2#0, %2#1) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   %4 = ccirc.xor(%3, %2#2) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   %5 = ccirc.xor(%4, %2#3) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   ccirc.return %5 : !ccirc<wirebundle[1]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @bv_oracle__xor[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0:5 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#0]:X %0#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%0#1]:X %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2, %result_3 = qcirc.gate1q[%0#3]:X %result_1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults, %controlResults_0, %0#2, %controlResults_2, %result_3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @bv_oracle__sign[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   %1 = qwerty.qbinit %0 as {list:{"|m>"}} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   %2 = qwerty.qbunpack %1 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
//  CHECK-NEXT:   %3:4 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qwerty.qbpack(%3#0, %3#1, %3#2, %3#3, %2) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %5 = qwerty.call @bv_oracle__xor(%4) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %6:5 = qwerty.qbunpack %5 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %7 = qwerty.qbpack(%6#0, %6#1, %6#2, %6#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %8 = qwerty.qbpack(%6#4) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   %9 = qwerty.qbdeinit %8 as {list:{"|m>"}} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   qwerty.qbdiscardz %9 : (!qwerty<qbundle[1]>) -> ()
//  CHECK-NEXT:   qwerty.return %7 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @embed_sign[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @bv_oracle__sign[] : () -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
ccirc.circuit private @bv_oracle(%arg0: !ccirc<wirebundle[4]>) irrev {
  // sign extended: 0b1101 | -1 << 4 == -3
  %0 = ccirc.constant -3 : i4 : !ccirc<wirebundle[4]>
  %1 = ccirc.and(%arg0, %0) : (!ccirc<wirebundle[4]>, !ccirc<wirebundle[4]>) -> !ccirc<wirebundle[4]>
  %2:4 = ccirc.wireunpack %1 : (!ccirc<wirebundle[4]>) -> (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>)
  %3 = ccirc.xor(%2#0, %2#1) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
  %4 = ccirc.xor(%3, %2#2) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
  %5 = ccirc.xor(%4, %2#3) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
  ccirc.return %5: !ccirc<wirebundle[1]>
}

qwerty.func @embed_sign[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.embed_sign @bv_oracle : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %1 : !qwerty<qbundle[4]>
}
