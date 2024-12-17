// RUN: qwerty-opt -test-qubit-index %s | FileCheck %s

// CHECK-LABEL: test_tag: trivial__ret
//  CHECK-NEXT:  operand #0: [0,1,2,3]
qwerty.func private @trivial[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  qwerty.return {tag = "trivial__ret"} %arg0 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: test_tag: implicit_swap__unpack:
//  CHECK-NEXT:  operand #0: [0,1]
//  CHECK-NEXT:  result #0: [0]
//  CHECK-NEXT:  result #1: [1]
//  CHECK-NEXT: test_tag: pack:
//  CHECK-NEXT:  operand #0: [1]
//  CHECK-NEXT:  operand #1: [0]
//  CHECK-NEXT:  result #0: [1,0]
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: [1,0]
qwerty.func private @implicit_swap[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0:2 = qwerty.qbunpack %arg0 {tag = "implicit_swap__unpack"} : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
  %1 = qwerty.qbpack(%0#1, %0#0) {tag = "pack"} : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
  qwerty.return {tag = "ret"} %1 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: test_tag: implicit_swap3__unpack:
//  CHECK-NEXT:  operand #0: [0,1,2]
//  CHECK-NEXT:  result #0: [0]
//  CHECK-NEXT:  result #1: [1]
//  CHECK-NEXT:  result #2: [2]
//  CHECK-NEXT: test_tag: pack:
//  CHECK-NEXT:  operand #0: [2]
//  CHECK-NEXT:  operand #1: [0]
//  CHECK-NEXT:  operand #2: [1]
//  CHECK-NEXT:  result #0: [2,0,1]
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: [2,0,1]
qwerty.func private @implicit_swap3[](%arg0: !qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]> {
  %0:3 = qwerty.qbunpack %arg0 {tag = "implicit_swap3__unpack"} : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
  %1 = qwerty.qbpack(%0#2, %0#0, %0#1) {tag = "pack"} : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
  qwerty.return {tag = "ret"} %1 : !qwerty<qbundle[3]>
}

// CHECK-LABEL: test_tag: freez__qalloc:
//  CHECK-NEXT:  result #0: [3]
//  CHECK-NEXT: test_tag: unpack:
//  CHECK-NEXT:  operand #0: [0,1,2]
//  CHECK-NEXT:  result #0: [0]
//  CHECK-NEXT:  result #1: [1]
//  CHECK-NEXT:  result #2: [2]
//  CHECK-NEXT: test_tag: pack:
//  CHECK-NEXT:  operand #0: [2]
//  CHECK-NEXT:  operand #1: [0]
//  CHECK-NEXT:  operand #2: [1]
//  CHECK-NEXT:  result #0: [2,0,1]
//  CHECK-NEXT: test_tag: freez:
//  CHECK-NEXT:  operand #0: [3]
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: [2,0,1]
qwerty.func private @freez[](%arg0: !qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]> {
  %0 = qcirc.qalloc {tag = "freez__qalloc"} : () -> !qcirc.qubit
  %1:3 = qwerty.qbunpack %arg0 {tag = "unpack"} : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
  %2 = qwerty.qbpack(%1#2, %1#0, %1#1) {tag = "pack"} : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
  qcirc.qfreez %0 {tag = "freez"} : (!qcirc.qubit) -> ()
  qwerty.return {tag = "ret"} %2 : !qwerty<qbundle[3]>
}

// CHECK-LABEL: test_tag: discardz__qbprep:
//  CHECK-NEXT:  result #0: [3,4,5,6,7]
//  CHECK-NEXT: test_tag: unpack:
//  CHECK-NEXT:  operand #0: [0,1,2]
//  CHECK-NEXT:  result #0: [0]
//  CHECK-NEXT:  result #1: [1]
//  CHECK-NEXT:  result #2: [2]
//  CHECK-NEXT: test_tag: pack:
//  CHECK-NEXT:  operand #0: [2]
//  CHECK-NEXT:  operand #1: [0]
//  CHECK-NEXT:  operand #2: [1]
//  CHECK-NEXT:  result #0: [2,0,1]
//  CHECK-NEXT: test_tag: discardz:
//  CHECK-NEXT:  operand #0: [3,4,5,6,7]
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: [2,0,1]
qwerty.func private @discardz[](%arg0: !qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]> {
  %0 = qwerty.qbprep {tag = "discardz__qbprep"} Z<PLUS>[5] : () -> !qwerty<qbundle[5]>
  %1:3 = qwerty.qbunpack %arg0 {tag = "unpack"} : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
  %2 = qwerty.qbpack(%1#2, %1#0, %1#1) {tag = "pack"} : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
  qwerty.qbdiscardz %0 {tag = "discardz"} : (!qwerty<qbundle[5]>) -> ()
  qwerty.return {tag = "ret"} %2 : !qwerty<qbundle[3]>
}

// CHECK-LABEL: test_tag: btrans__unpack:
//  CHECK-NEXT:  operand #0: [0,1]
//  CHECK-NEXT:  result #0: [0]
//  CHECK-NEXT:  result #1: [1]
//  CHECK-NEXT: test_tag: pack:
//  CHECK-NEXT:  operand #0: [1]
//  CHECK-NEXT:  operand #1: [0]
//  CHECK-NEXT:  result #0: [1,0]
//  CHECK-NEXT: test_tag: btrans:
//  CHECK-NEXT:  operand #0: [1,0]
//  CHECK-NEXT:  result #0: [1,0]
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: [1,0]
qwerty.func private @btrans[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0:2 = qwerty.qbunpack %arg0 {tag = "btrans__unpack"} : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
  %1 = qwerty.qbpack(%0#1, %0#0) {tag = "pack"} : (!qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[2]>
  %2 = qwerty.qbtrans %1 by {std:Z[2]} >> {std:X[2]} {tag = "btrans"} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return {tag = "ret"} %2 : !qwerty<qbundle[2]>
}
