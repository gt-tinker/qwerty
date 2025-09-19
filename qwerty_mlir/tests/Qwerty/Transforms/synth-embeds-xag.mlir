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

// CHECK-LABEL: qwerty.func private @silly__xor[](%arg0: !qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]> {
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
ccirc.circuit private @silly(%arg0: !ccirc<wire[4]>) rev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.not(%0#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %2 = ccirc.not(%0#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.not(%0#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %4 = ccirc.not(%0#3) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %5 = ccirc.wirepack(%1, %2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
  ccirc.return %4, %5 : !ccirc<wire[1]>, !ccirc<wire[3]>
}

qwerty.func @embed_xor[](%arg0: !qwerty<qbundle[8]>) irrev-> !qwerty<qbundle[8]> {
  %0 = qwerty.embed_xor @silly : !qwerty<func(!qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]>>, !qwerty<qbundle[8]>) -> !qwerty<qbundle[8]>
  qwerty.return %1 : !qwerty<qbundle[8]>
}

// -----

// CHECK-LABEL: qwerty.func private @and__xor[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0:5 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults:2, %result = qcirc.gate1q[%0#0, %0#1]:X %0#4 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0:2, %result_1 = qcirc.gate1q[%0#2, %0#3]:X %result : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults#0, %controlResults#1, %controlResults_0#0, %controlResults_0#1, %result_1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
ccirc.circuit private @and(%arg0: !ccirc<wire[4]>) irrev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.and(%0#0, %0#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %2 = ccirc.and(%0#2, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.parity(%1, %2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %3 : !ccirc<wire[1]>
}

qwerty.func @embed_xor[](%arg0: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
  %0 = qwerty.embed_xor @and : !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %1 : !qwerty<qbundle[5]>
}

// -----

// CHECK-LABEL: qwerty.func private @and_and__xor[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0:5 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults:2, %result = qcirc.gate1q[%0#0, %0#1]:X %1 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_0:2, %result_1 = qcirc.gate1q[%0#2, %0#3]:X %2 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2:2, %result_3 = qcirc.gate1q[%result, %result_1]:X %0#4 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_4:2, %result_5 = qcirc.gate1q[%controlResults_0#0, %controlResults_0#1]:X %controlResults_2#1 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_5 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_6:2, %result_7 = qcirc.gate1q[%controlResults#0, %controlResults#1]:X %controlResults_2#0 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_7 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %3 = qwerty.qbpack(%controlResults_6#0, %controlResults_6#1, %controlResults_4#0, %controlResults_4#1, %result_3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %3 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
ccirc.circuit private @and_and(%arg0: !ccirc<wire[4]>) irrev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.and(%0#0, %0#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %2 = ccirc.and(%0#2, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.and(%1, %2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %3 : !ccirc<wire[1]>
}

qwerty.func @embed_xor[](%arg0: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
  %0 = qwerty.embed_xor @and_and : !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %1 : !qwerty<qbundle[5]>
}

// -----

// CHECK-LABEL: qwerty.func private @and_not_and__xor[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0:5 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults:2, %result = qcirc.gate1q[%0#0, %0#1]:X %1 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:X %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_1:2, %result_2 = qcirc.gate1q[%0#2, %0#3]:X %2 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_3:2, %result_4 = qcirc.gate1q[%result_0, %result_2]:X %0#4 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_5:2, %result_6 = qcirc.gate1q[%controlResults_1#0, %controlResults_1#1]:X %controlResults_3#1 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_6 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %result_7 = qcirc.gate1q[]:X %controlResults_3#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_8:2, %result_9 = qcirc.gate1q[%controlResults#0, %controlResults#1]:X %result_7 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_9 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %3 = qwerty.qbpack(%controlResults_8#0, %controlResults_8#1, %controlResults_5#0, %controlResults_5#1, %result_4) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %3 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
ccirc.circuit private @and_not_and(%arg0: !ccirc<wire[4]>) irrev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.and(%0#0, %0#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %2 = ccirc.not(%1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.and(%0#2, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %4 = ccirc.and(%2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %4 : !ccirc<wire[1]>
}

qwerty.func @embed_xor[](%arg0: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
  %0 = qwerty.embed_xor @and_not_and : !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %1 : !qwerty<qbundle[5]>
}
