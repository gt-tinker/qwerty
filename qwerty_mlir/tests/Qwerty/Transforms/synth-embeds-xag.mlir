// RUN: qwerty-opt -split-input-file -synth-embeds %s | FileCheck %s

// CHECK-LABEL: ccirc.circuit private @trivial(%arg0: !ccirc<wire[1]>) rev {
//  CHECK-NEXT:   ccirc.return %arg0 : !ccirc<wire[1]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @trivial__xor[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0:2 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#0]:X %0#1 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults, %result) : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @embed_xor1[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @trivial__xor[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
ccirc.circuit private @trivial(%arg0: !ccirc<wire[1]>) rev {
  ccirc.return %arg0 : !ccirc<wire[1]>
}

qwerty.func @embed_xor1[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<qbundle[2]> {
  %0 = qwerty.embed_xor @trivial : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// -----

// CHECK-LABEL: ccirc.circuit private @flip(%arg0: !ccirc<wire[1]>) rev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %0 : !ccirc<wire[1]>
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
ccirc.circuit private @flip(%arg0: !ccirc<wire[1]>) rev {
  %0 = ccirc.not(%arg0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %0 : !ccirc<wire[1]>
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

// CHECK-LABEL: ccirc.circuit private @bv_oracle(%arg0: !ccirc<wire[4]>) irrev {
//  CHECK-NEXT:   %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1 = ccirc.parity(%0#0, %0#1, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %1 : !ccirc<wire[1]>
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
ccirc.circuit private @bv_oracle(%arg0: !ccirc<wire[4]>) irrev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.parity(%0#0, %0#1, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %1 : !ccirc<wire[1]>
}

qwerty.func @embed_sign[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.embed_sign @bv_oracle : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %1 : !qwerty<qbundle[4]>
}

// -----
// CHECK-LABEL: ccirc.circuit private @silly(%arg0: !ccirc<wire[4]>) rev {
//  CHECK-NEXT:   %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1 = ccirc.not(%0#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %2 = ccirc.not(%0#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.not(%0#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.not(%0#3) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %5 = ccirc.wirepack(%1, %2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
//  CHECK-NEXT:   ccirc.return %4, %5 : !ccirc<wire[1]>, !ccirc<wire[3]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @silly__xor[](%arg0: !qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]> {
//  CHECK-NEXT:   %0:8 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#3]:X %0#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:X %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_1, %result_2 = qcirc.gate1q[%0#1]:X %0#5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:X %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_4, %result_5 = qcirc.gate1q[%0#0]:X %0#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_7, %result_8 = qcirc.gate1q[%0#2]:X %0#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:X %result_8 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults_4, %controlResults_1, %controlResults_7, %controlResults, %result_0, %result_3, %result_6, %result_9) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[8]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @embed_inplace[](%arg0: !qwerty<qbundle[8]>) irrev-> !qwerty<qbundle[8]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @silly__xor[] : () -> !qwerty<func(!qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]>>, !qwerty<qbundle[8]>) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[8]>
//  CHECK-NEXT: }
ccirc.circuit private @silly(%arg0: !ccirc<wire[4]>) rev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.not(%0#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %2 = ccirc.not(%0#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.not(%0#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %4 = ccirc.not(%0#3) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %5 = ccirc.wirepack(%1, %2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
  ccirc.return %4, %5 : !ccirc<wire[1]>, !ccirc<wire[3]>
}

qwerty.func @embed_inplace[](%arg0: !qwerty<qbundle[8]>) irrev-> !qwerty<qbundle[8]> {
  %0 = qwerty.embed_xor @silly : !qwerty<func(!qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]>>, !qwerty<qbundle[8]>) -> !qwerty<qbundle[8]>
  qwerty.return %1 : !qwerty<qbundle[8]>
}
