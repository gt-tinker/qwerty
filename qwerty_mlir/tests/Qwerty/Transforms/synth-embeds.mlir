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

// -----

// CHECK-LABEL: ccirc.circuit private @silly(%arg0: !ccirc<wirebundle[4]>) rev {
//  CHECK-NEXT:   %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wirebundle[4]>) -> (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>)
//  CHECK-NEXT:   %1 = ccirc.wirepack(%0#0, %0#2) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[2]>
//  CHECK-NEXT:   %2 = ccirc.wirepack(%0#1, %1) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[2]>) -> !ccirc<wirebundle[3]>
//  CHECK-NEXT:   %3 = ccirc.not(%2) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
//  CHECK-NEXT:   %4 = ccirc.not(%0#3) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   ccirc.return %4, %3 : !ccirc<wirebundle[1]>, !ccirc<wirebundle[3]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @silly__xor[](%arg0: !qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]> {
//  CHECK-NEXT:   %0:8 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#3]:X %0#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%0#1]:X %0#5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2, %result_3 = qcirc.gate1q[%0#0]:X %0#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_4, %result_5 = qcirc.gate1q[%0#2]:X %0#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:X %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_7 = qcirc.gate1q[]:X %result_1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_8 = qcirc.gate1q[]:X %result_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults_2, %controlResults_0, %controlResults_4, %controlResults, %result_6, %result_7, %result_8, %result_9) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[8]>
//  CHECK-NEXT: }
//  CHECK-NEXT: ccirc.circuit private @silly__inv(%arg0: !ccirc<wirebundle[1]>, %arg1: !ccirc<wirebundle[3]>) rev {
//  CHECK-NEXT:   %0 = ccirc.not(%arg0) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   %1 = ccirc.not(%arg1) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
//  CHECK-NEXT:   %2:3 = ccirc.wireunpack %1 : (!ccirc<wirebundle[3]>) -> (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>)
//  CHECK-NEXT:   %3 = ccirc.wirepack(%2#0) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   %4 = ccirc.wirepack(%2#1, %2#2) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[2]>
//  CHECK-NEXT:   %5:2 = ccirc.wireunpack %4 : (!ccirc<wirebundle[2]>) -> (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>)
//  CHECK-NEXT:   %6 = ccirc.wirepack(%5#0) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   %7 = ccirc.wirepack(%5#1) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
//  CHECK-NEXT:   %8 = ccirc.wirepack(%6, %3, %7, %0) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[4]>
//  CHECK-NEXT:   ccirc.return %8 : !ccirc<wirebundle[4]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @silly__inv__xor[](%arg0: !qwerty<qbundle[8]>) rev-> !qwerty<qbundle[8]> {
//  CHECK-NEXT:   %0:8 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults, %result = qcirc.gate1q[%0#2]:X %0#4 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_0, %result_1 = qcirc.gate1q[%0#1]:X %0#5 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_2, %result_3 = qcirc.gate1q[%0#3]:X %0#6 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %controlResults_4, %result_5 = qcirc.gate1q[%0#0]:X %0#7 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %result_6 = qcirc.gate1q[]:X %result : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_7 = qcirc.gate1q[]:X %result_1 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_8 = qcirc.gate1q[]:X %result_3 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %result_9 = qcirc.gate1q[]:X %result_5 : (!qcirc.qubit) -> !qcirc.qubit
//  CHECK-NEXT:   %1 = qwerty.qbpack(%controlResults_4, %controlResults_0, %controlResults, %controlResults_2, %result_6, %result_7, %result_8, %result_9) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[8]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qwerty.func private @silly__inplace[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbprep Z<PLUS>[4] : () -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %1:4 = qwerty.qbunpack %0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %2:4 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %3 = qwerty.qbpack(%2#0, %2#1, %2#2, %2#3, %1#0, %1#1, %1#2, %1#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   %4 = qwerty.call @silly__xor(%3) : (!qwerty<qbundle[8]>) -> !qwerty<qbundle[8]>
//  CHECK-NEXT:   %5:8 = qwerty.qbunpack %4 : (!qwerty<qbundle[8]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
//  CHECK-NEXT:   %6 = qwerty.qbpack(%5#4, %5#5, %5#6, %5#7, %5#0, %5#1, %5#2, %5#3) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[8]>
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
ccirc.circuit private @silly(%arg0: !ccirc<wirebundle[4]>) rev {
  %0:4 = ccirc.wireunpack %arg0 : (!ccirc<wirebundle[4]>) -> (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>)
  %1 = ccirc.wirepack(%0#0, %0#2) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[1]>) -> !ccirc<wirebundle[2]>
  %2 = ccirc.wirepack(%0#1, %1) : (!ccirc<wirebundle[1]>, !ccirc<wirebundle[2]>) -> !ccirc<wirebundle[3]>
  %3 = ccirc.not(%2) : (!ccirc<wirebundle[3]>) -> !ccirc<wirebundle[3]>
  %4 = ccirc.not(%0#3) : (!ccirc<wirebundle[1]>) -> !ccirc<wirebundle[1]>
  ccirc.return %4, %3 : !ccirc<wirebundle[1]>, !ccirc<wirebundle[3]>
}

qwerty.func @embed_inplace[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.embed_inplace @silly : !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %1 : !qwerty<qbundle[4]>
}
