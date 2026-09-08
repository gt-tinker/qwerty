// RUN: qwerty-opt -split-input-file -canonicalize -synth-embeds %s | FileCheck %s

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
  // sign extended: 0b1101 | -1 << 4 == -3
  %0 = ccirc.constant -3 : i4 : !ccirc<wire[4]>
  %1 = ccirc.and(%arg0, %0) : (!ccirc<wire[4]>, !ccirc<wire[4]>) -> !ccirc<wire[4]>
  %2:4 = ccirc.wireunpack %1 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %3 = ccirc.xor(%2#0, %2#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %4 = ccirc.xor(%3, %2#2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %5 = ccirc.xor(%4, %2#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %5: !ccirc<wire[1]>
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
//  CHECK-NEXT:   %4 = ccirc.wirepack(%1, %2, %3) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[3]>
//  CHECK-NEXT:   %5 = ccirc.not(%0#3) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %5, %4 : !ccirc<wire[1]>, !ccirc<wire[3]>
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
//  CHECK-NEXT: ccirc.circuit private @silly__inv(%arg0: !ccirc<wire[1]>, %arg1: !ccirc<wire[3]>) rev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %1:3 = ccirc.wireunpack %arg1 : (!ccirc<wire[3]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %2 = ccirc.not(%1#0) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.not(%1#1) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %4 = ccirc.not(%1#2) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %5 = ccirc.wirepack(%3, %2, %4, %0) : (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[4]>
//  CHECK-NEXT:   ccirc.return %5 : !ccirc<wire[4]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @silly__inv__xor[](%arg0: !qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]> {
//  CHECK-NEXT:   %0:8 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#2]:X %0#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_0 = qcirc.gate1q[]:X %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_1, %result_2 = qcirc.gate1q[%0#1]:X %0#5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_3 = qcirc.gate1q[]:X %result_2 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_4, %result_5 = qcirc.gate1q[%0#3]:X %0#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_7, %result_8 = qcirc.gate1q[%0#0]:X %0#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:X %result_8 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults_7, %controlResults_1, %controlResults, %controlResults_4, %result_0, %result_3, %result_6, %result_9) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[8]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @silly__inplace[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[4] : () -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %1:4 = qwerty.qbunpack %0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %2:4 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %3 = qwerty.qbpack(%2#0, %2#1, %2#2, %2#3, %1#0, %1#1, %1#2, %1#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   %4 = qwerty.call @silly__xor(%3) : (!qwerty<qbundle[8]>) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   %5:8 = qwerty.qbunpack %4 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %leftResult, %rightResult = qcirc.gate2q[]:Swap %5#0, %5#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %leftResult_0, %rightResult_1 = qcirc.gate2q[]:Swap %5#1, %5#5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %leftResult_2, %rightResult_3 = qcirc.gate2q[]:Swap %5#2, %5#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %leftResult_4, %rightResult_5 = qcirc.gate2q[]:Swap %5#3, %5#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %6 = qwerty.qbpack(%leftResult, %leftResult_0, %leftResult_2, %leftResult_4, %rightResult, %rightResult_1, %rightResult_3, %rightResult_5) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   %7 = qwerty.call @silly__inv__xor(%6) : (!qwerty<qbundle[8]>) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   %8:8 = qwerty.qbunpack %7 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %9 = qwerty.qbpack(%8#4, %8#5, %8#6, %8#7) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.qbdiscardz %9 : (!qwerty<qbundle[4]>) -> ()
//  CHECK-NEXT:   %10 = qwerty.qbpack(%8#0, %8#1, %8#2, %8#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.return %10 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @embed_inplace[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @silly__inplace[] : () -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
ccirc.circuit private @silly(%arg0: !ccirc<wire[4]>) rev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.wirepack(%0#0, %0#2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[2]>
  %2 = ccirc.wirepack(%0#1, %1) : (!ccirc<wire[1]>, !ccirc<wire[2]>) -> !ccirc<wire[3]>
  %3 = ccirc.not(%2) : (!ccirc<wire[3]>) -> !ccirc<wire[3]>
  %4 = ccirc.not(%0#3) : (!ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %4, %3 : !ccirc<wire[1]>, !ccirc<wire[3]>
}

qwerty.func @embed_inplace[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.embed_inplace @silly : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %1 : !qwerty<qbundle[4]>
}

// -----

// CHECK-LABEL: ccirc.circuit private @all_ones_0(%arg0: !ccirc<wire[4]>) irrev {
//  CHECK-NEXT:   %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
//  CHECK-NEXT:   %1 = ccirc.and(%0#0, %0#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %2 = ccirc.and(%1, %0#2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   %3 = ccirc.and(%2, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
//  CHECK-NEXT:   ccirc.return %3 : !ccirc<wire[1]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @all_ones_0__xor[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0:5 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %3 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant 3.1415926535897931 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %controlResults:2, %result = qcirc.gate1q1p[%0#0, %0#1]:Rx(%3) %2 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0:2, %result_1 = qcirc.gate1q1p[%result, %0#2]:Rx(%3) %1 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2:2, %result_3 = qcirc.gate1q[%result_1, %0#3]:X %0#4 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant -3.1415926535897931 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %controlResults_4:2, %result_5 = qcirc.gate1q1p[%controlResults_0#0, %controlResults_0#1]:Rx(%4) %controlResults_2#0 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_6:2, %result_7 = qcirc.gate1q1p[%controlResults#0, %controlResults#1]:Rx(%4) %controlResults_4#0 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_7 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_5 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %5 = qwerty.qbpack(%controlResults_6#0, %controlResults_6#1, %controlResults_4#1, %controlResults_2#1, %result_3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %5 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @all_ones_0__sign[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   %1 = qwerty.qbinit %0 as {list:{"|m>"}} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   %2 = qwerty.qbunpack %1 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
//  CHECK-NEXT:   %3:4 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qwerty.qbpack(%3#0, %3#1, %3#2, %3#3, %2) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %5 = qwerty.call @all_ones_0__xor(%4) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   %6:5 = qwerty.qbunpack %5 : (!qwerty<qbundle[5]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %7 = qwerty.qbpack(%6#0, %6#1, %6#2, %6#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %8 = qwerty.qbpack(%6#4) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   %9 = qwerty.qbdeinit %8 as {list:{"|m>"}} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:   qwerty.qbdiscardz %9 : (!qwerty<qbundle[1]>) -> ()
//  CHECK-NEXT:   qwerty.return %7 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func @grover_iter_1[]() irrev-> !qwerty<bitbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[4] : () -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %1 = qwerty.func_const @all_ones_0__sign[] : () -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
//  CHECK-NEXT:   %2 = qwerty.call_indirect %1(%0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %3 = qwerty.qbmeas %2 by {std: Z[4]} : !qwerty<qbundle[4]> -> !qwerty<bitbundle[4]>
//  CHECK-NEXT:   qwerty.return %3 : !qwerty<bitbundle[4]>
//  CHECK-NEXT: }
ccirc.circuit private @all_ones_0(%arg0: !ccirc<wire[4]>) irrev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wire[4]>) -> (!ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>, !ccirc<wire[1]>)
  %1 = ccirc.and(%0#0, %0#1) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %2 = ccirc.and(%1, %0#2) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  %3 = ccirc.and(%2, %0#3) : (!ccirc<wire[1]>, !ccirc<wire[1]>) -> !ccirc<wire[1]>
  ccirc.return %3 : !ccirc<wire[1]>
}
qwerty.func @grover_iter_1[]() irrev-> !qwerty<bitbundle[4]> {
  %0 = qwerty.qbprep Z<PLUS>[4] : () -> !qwerty<qbundle[4]>
  %1 = qwerty.embed_sign @all_ones_0 : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %2 = qwerty.call_indirect %1(%0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  %3 = qwerty.qbmeas %2 by {std: Z[4]} : !qwerty<qbundle[4]> -> !qwerty<bitbundle[4]>
  qwerty.return %3 : !qwerty<bitbundle[4]>
}

// -----

// CHECK-LABEL: qwerty.func private @flip__xor[](%arg0: !qwerty<qbundle[9]>) rev-> !qwerty<qbundle[9]> {
//  CHECK-NEXT:   %0:9 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[9]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#0]:X %0#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%0#3]:X %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %3 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant 3.1415926535897931 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %controlResults_2:2, %result_3 = qcirc.gate1q1p[%0#2, %0#5]:Rx(%3) %2 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_4:2, %result_5 = qcirc.gate1q1p[%0#4, %result_3]:Rx(%3) %1 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %4 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %5 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_6:2, %result_7 = qcirc.gate1q1p[%0#1, %controlResults_4#0]:Rx(%3) %5 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %6 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_8:2, %result_9 = qcirc.gate1q1p[%controlResults_6#0, %controlResults_4#1]:Rx(%3) %6 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_10 = qcirc.gate1q[]:X %result_7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_11 = qcirc.gate1q[]:X %result_9 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_12:2, %result_13 = qcirc.gate1q1p[%result_10, %result_11]:Rx(%3) %4 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_14 = qcirc.gate1q[]:X %controlResults_12#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_15 = qcirc.gate1q[]:X %controlResults_12#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_16 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_17:2, %result_18 = qcirc.gate1q[%result_16, %result_13]:X %result_1 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_19 = qcirc.gate1q[]:X %controlResults_17#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_20 = qcirc.gate1q[]:X %result_15 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_21 = qcirc.gate1q[]:X %result_14 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %7 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant -3.1415926535897931 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %controlResults_22:2, %result_23 = qcirc.gate1q1p[%result_20, %result_21]:Rx(%7) %controlResults_17#1 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_24 = qcirc.gate1q[]:X %controlResults_22#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_25 = qcirc.gate1q[]:X %controlResults_22#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_26:2, %result_27 = qcirc.gate1q1p[%controlResults_8#0, %controlResults_8#1]:Rx(%7) %result_24 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_27 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_28:2, %result_29 = qcirc.gate1q1p[%controlResults_26#0, %controlResults_6#1]:Rx(%7) %result_25 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_29 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_23 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_30:2, %result_31 = qcirc.gate1q1p[%controlResults_28#1, %controlResults_26#1]:Rx(%7) %result_19 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_31 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %result_32 = qcirc.gate1q[]:X %result_18 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_33, %result_34 = qcirc.gate1q[%controlResults_28#0]:X %0#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_35, %result_36 = qcirc.gate1q[%controlResults_30#0]:X %result_34 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_37, %result_38 = qcirc.gate1q[%controlResults_30#1]:X %result_36 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_39:2, %result_40 = qcirc.gate1q1p[%controlResults_2#0, %controlResults_2#1]:Rx(%7) %controlResults_37 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_40 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_41, %result_42 = qcirc.gate1q[%controlResults_39#0]:X %0#8 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_43, %result_44 = qcirc.gate1q[%controlResults_39#1]:X %result_42 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %8 = qwerty.qbpack(%controlResults, %controlResults_33, %controlResults_41, %controlResults_0, %controlResults_35, %controlResults_43, %result_32, %result_38, %result_44) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[9]>
//  CHECK-NEXT:   qwerty.return %8 : !qwerty<qbundle[9]>
//  CHECK-NEXT: }
ccirc.circuit private @flip(%arg0: !ccirc<wire[3]>, %arg1: !ccirc<wire[3]>) irrev {
  %0 = ccirc.add(%arg0, %arg1) : (!ccirc<wire[3]>, !ccirc<wire[3]>) -> !ccirc<wire[3]>
  ccirc.return %0 : !ccirc<wire[3]>
}

qwerty.func @embed_xor1[](%arg0: !qwerty<qbundle[9]>) irrev-> !qwerty<qbundle[9]> {
  %0 = qwerty.embed_xor @flip : !qwerty<func(!qwerty<qbundle[9]>) rev-> !qwerty<qbundle[9]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[9]>) rev-> !qwerty<qbundle[9]>>, !qwerty<qbundle[9]>) -> !qwerty<qbundle[9]>
  qwerty.return %1 : !qwerty<qbundle[9]>
}

// -----

// CHECK-LABEL: qwerty.func private @flip__xor[](%arg0: !qwerty<qbundle[12]>) rev-> !qwerty<qbundle[12]> {
//  CHECK-NEXT:   %0:12 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[12]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#0]:X %0#8 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%0#4]:X %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %1 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %2 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %3 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %4 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %5 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant 3.1415926535897931 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %controlResults_2:2, %result_3 = qcirc.gate1q1p[%0#3, %0#7]:Rx(%5) %4 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_4:2, %result_5 = qcirc.gate1q1p[%0#6, %result_3]:Rx(%5) %3 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %6 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %7 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_6:2, %result_7 = qcirc.gate1q1p[%0#2, %controlResults_4#0]:Rx(%5) %7 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %8 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_8:2, %result_9 = qcirc.gate1q1p[%controlResults_6#0, %controlResults_4#1]:Rx(%5) %8 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_10 = qcirc.gate1q[]:X %result_7 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_11 = qcirc.gate1q[]:X %result_9 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_12:2, %result_13 = qcirc.gate1q1p[%result_10, %result_11]:Rx(%5) %6 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_14 = qcirc.gate1q[]:X %controlResults_12#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_15 = qcirc.gate1q[]:X %controlResults_12#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_16 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_17:2, %result_18 = qcirc.gate1q1p[%result_16, %result_13]:Rx(%5) %2 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_19 = qcirc.gate1q[]:X %controlResults_17#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_20 = qcirc.gate1q[]:X %result_18 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_21:2, %result_22 = qcirc.gate1q1p[%0#5, %result_20]:Rx(%5) %1 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_23 = qcirc.gate1q[]:X %controlResults_21#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %9 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %10 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_24:2, %result_25 = qcirc.gate1q1p[%0#1, %controlResults_21#0]:Rx(%5) %10 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %11 = qcirc.qalloc : () -> !qcirc.qubit
//  CHECK-NEXT:   %result_26 = qcirc.gate1q[]:X %result_23 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_27:2, %result_28 = qcirc.gate1q1p[%controlResults_24#0, %result_26]:Rx(%5) %11 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_29 = qcirc.gate1q[]:X %controlResults_27#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_30 = qcirc.gate1q[]:X %result_25 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_31 = qcirc.gate1q[]:X %result_28 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_32:2, %result_33 = qcirc.gate1q1p[%result_30, %result_31]:Rx(%5) %9 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_34 = qcirc.gate1q[]:X %controlResults_32#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_35 = qcirc.gate1q[]:X %controlResults_32#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_36 = qcirc.gate1q[]:X %result_22 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_37:2, %result_38 = qcirc.gate1q[%result_36, %result_33]:X %result_1 : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_39 = qcirc.gate1q[]:X %controlResults_37#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_40 = qcirc.gate1q[]:X %result_35 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_41 = qcirc.gate1q[]:X %result_34 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %12 = qcirc.calc() : () -> f64 {
//  CHECK-NEXT:     %cst = arith.constant -3.1415926535897931 : f64
//  CHECK-NEXT:     qcirc.calc_yield(%cst) : f64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   %controlResults_42:2, %result_43 = qcirc.gate1q1p[%result_40, %result_41]:Rx(%12) %controlResults_37#1 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_44 = qcirc.gate1q[]:X %controlResults_42#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_45 = qcirc.gate1q[]:X %controlResults_42#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_46 = qcirc.gate1q[]:X %result_29 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_47:2, %result_48 = qcirc.gate1q1p[%controlResults_27#0, %result_46]:Rx(%12) %result_44 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_49 = qcirc.gate1q[]:X %controlResults_47#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   qcirc.qfreez %result_48 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_50:2, %result_51 = qcirc.gate1q1p[%controlResults_47#0, %controlResults_24#1]:Rx(%12) %result_45 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_51 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_43 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %result_52 = qcirc.gate1q[]:X %result_49 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_53:2, %result_54 = qcirc.gate1q1p[%controlResults_50#1, %result_52]:Rx(%12) %result_39 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_55 = qcirc.gate1q[]:X %controlResults_53#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   qcirc.qfreez %result_54 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %result_56 = qcirc.gate1q[]:X %result_38 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_57, %result_58 = qcirc.gate1q[%controlResults_50#0]:X %0#9 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_59, %result_60 = qcirc.gate1q[%controlResults_53#0]:X %result_58 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_61, %result_62 = qcirc.gate1q[%result_55]:X %result_60 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_63 = qcirc.gate1q[]:X %result_19 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_64:2, %result_65 = qcirc.gate1q1p[%result_63, %controlResults_17#1]:Rx(%12) %controlResults_61 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_66 = qcirc.gate1q[]:X %controlResults_64#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_67 = qcirc.gate1q[]:X %result_15 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_68 = qcirc.gate1q[]:X %result_14 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_69:2, %result_70 = qcirc.gate1q1p[%result_67, %result_68]:Rx(%12) %controlResults_64#1 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_71 = qcirc.gate1q[]:X %controlResults_69#1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_72 = qcirc.gate1q[]:X %controlResults_69#0 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_73:2, %result_74 = qcirc.gate1q1p[%controlResults_8#0, %controlResults_8#1]:Rx(%12) %result_71 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_74 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_75:2, %result_76 = qcirc.gate1q1p[%controlResults_73#0, %controlResults_6#1]:Rx(%12) %result_72 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_76 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_70 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_77:2, %result_78 = qcirc.gate1q1p[%controlResults_75#1, %controlResults_73#1]:Rx(%12) %result_66 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_78 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   qcirc.qfreez %result_65 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %result_79 = qcirc.gate1q[]:X %result_62 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %controlResults_80, %result_81 = qcirc.gate1q[%controlResults_75#0]:X %0#10 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_82, %result_83 = qcirc.gate1q[%controlResults_77#0]:X %result_81 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_84, %result_85 = qcirc.gate1q[%controlResults_77#1]:X %result_83 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_86:2, %result_87 = qcirc.gate1q1p[%controlResults_2#0, %controlResults_2#1]:Rx(%12) %controlResults_84 : (f64, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   qcirc.qfreez %result_87 : (!qcirc.qubit) -> ()
//  CHECK-NEXT:   %controlResults_88, %result_89 = qcirc.gate1q[%controlResults_86#0]:X %0#11 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_90, %result_91 = qcirc.gate1q[%controlResults_86#1]:X %result_89 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %13 = qwerty.qbpack(%controlResults, %controlResults_57, %controlResults_80, %controlResults_88, %controlResults_0, %controlResults_59, %controlResults_82, %controlResults_90, %result_56, %result_79, %result_85, %result_91) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[12]>
//  CHECK-NEXT:   qwerty.return %13 : !qwerty<qbundle[12]>
//  CHECK-NEXT: }
ccirc.circuit private @flip(%arg0: !ccirc<wire[4]>, %arg1: !ccirc<wire[4]>) irrev {
  %0 = ccirc.add(%arg0, %arg1) : (!ccirc<wire[4]>, !ccirc<wire[4]>) -> !ccirc<wire[4]>
  ccirc.return %0 : !ccirc<wire[4]>
}

qwerty.func @embed_xor1[](%arg0: !qwerty<qbundle[12]>) irrev-> !qwerty<qbundle[12]> {
  %0 = qwerty.embed_xor @flip : !qwerty<func(!qwerty<qbundle[12]>) rev-> !qwerty<qbundle[12]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[12]>) rev-> !qwerty<qbundle[12]>>, !qwerty<qbundle[12]>) -> !qwerty<qbundle[12]>
  qwerty.return %1 : !qwerty<qbundle[12]>
}
