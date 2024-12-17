// RUN: qwerty-opt -split-input-file -test-func-spec %s | FileCheck %s

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: test_tag: const_call__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @const_call[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @trivial[] {tag = "const_call__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call_indirect %0(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return {tag = "ret"} %1 : !qwerty<qbundle[2]>
}

// -----

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: test_tag: const_adj_call__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: func_adj:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,adj,0)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(trivial,adj,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @const_adj_call[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @trivial[] {tag = "const_adj_call__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 {tag = "func_adj"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.call_indirect %1(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return {tag = "ret"} %2 : !qwerty<qbundle[2]>
}

// -----

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: test_tag: const_adj_pred_call__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: func_adj:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,adj,0)}
//  CHECK-NEXT: test_tag: func_pred:
//  CHECK-NEXT:  operand #0: {(trivial,adj,0)}
//  CHECK-NEXT:  result #0: {(trivial,adj,2)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(trivial,adj,2)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @const_adj_pred_call[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_const @trivial[] {tag = "const_adj_pred_call__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 {tag = "func_adj"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.func_pred %1 by {list:{"|pp>"}} {tag = "func_pred"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %3 = qwerty.call_indirect %2(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "ret"} %3 : !qwerty<qbundle[4]>
}

// -----

// CHECK-LABEL: test_tag: calli_arg__func_adj:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: func_pred:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @calli_arg[](%arg0: !qwerty<qbundle[4]>, %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_adj %arg1 {tag = "calli_arg__func_adj"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|pp>"}} {tag = "func_pred"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %2 = qwerty.call_indirect %1(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "ret"} %2 : !qwerty<qbundle[4]>
}

// -----

// CHECK-LABEL: test_tag: priv_calli_arg__func_adj:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,adj,0)}
//  CHECK-NEXT: test_tag: func_pred:
//  CHECK-NEXT:  operand #0: {(trivial,adj,0)}
//  CHECK-NEXT:  result #0: {(trivial,adj,2)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(trivial,adj,2)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: callee_ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func private @priv_calli_arg[](%arg0: !qwerty<qbundle[4]>, %arg1: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_adj %arg1 {tag = "priv_calli_arg__func_adj"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|pp>"}} {tag = "func_pred"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %2 = qwerty.call_indirect %1(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "callee_ret"} %2 : !qwerty<qbundle[4]>
}

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: test_tag: call_with_const_arg__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: call:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @call_with_const_arg[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_const @trivial[] {tag = "call_with_const_arg__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call @priv_calli_arg(%arg0, %0) {tag = "call"} : (!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "ret"} %1 : !qwerty<qbundle[4]>
}

// -----

// CHECK-LABEL: test_tag: priv_calli_multi_arg__func_adj:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,2), (trivial4,adj,0), (trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,adj,2), (trivial4,fwd,0), (trivial4,adj,0)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(trivial,adj,2), (trivial4,fwd,0), (trivial4,adj,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: callee_ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func private @priv_calli_multi_arg[](%arg0: !qwerty<qbundle[4]>, %arg1: !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_adj %arg1 {tag = "priv_calli_multi_arg__func_adj"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "callee_ret"} %1 : !qwerty<qbundle[4]>
}

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

qwerty.func private @trivial4[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  qwerty.return %arg0 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: test_tag: multi_call_with_const_arg__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: func_pred:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,fwd,2)}
//  CHECK-NEXT: test_tag: call:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: {(trivial,fwd,2)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: func_const4:
//  CHECK-NEXT:  result #0: {(trivial4,fwd,0)}
//  CHECK-NEXT: test_tag: func_adj:
//  CHECK-NEXT:  operand #0: {(trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial4,adj,0)}
//  CHECK-NEXT: test_tag: call4_adj:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: {(trivial4,adj,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: call4:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: {(trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @multi_call_with_const_arg[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_const @trivial[] {tag = "multi_call_with_const_arg__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|pp>"}} {tag = "func_pred"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %2 = qwerty.call @priv_calli_multi_arg(%arg0, %1) {tag = "call"} : (!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  %3 = qwerty.func_const @trivial4[] {tag = "func_const4"} : () -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %4 = qwerty.func_adj %3 {tag = "func_adj"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %5 = qwerty.call @priv_calli_multi_arg(%2, %4) {tag = "call4_adj"} : (!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  %6 = qwerty.call @priv_calli_multi_arg(%5, %3) {tag = "call4"} : (!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "ret"} %6 : !qwerty<qbundle[4]>
}

// -----

// Unfortunate imprecision

// CHECK-LABEL: test_tag: priv_calli_multi_arg__func_adj:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: callee_ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func private @priv_calli_multi_arg[](%arg0: !qwerty<qbundle[4]>, %arg1: !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_adj %arg1 {tag = "priv_calli_multi_arg__func_adj"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "callee_ret"} %1 : !qwerty<qbundle[4]>
}

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

qwerty.func private @trivial4[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  qwerty.return %arg0 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: test_tag: multi_calli_with_const_arg__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: func_pred:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,fwd,2)}
//  CHECK-NEXT: test_tag: callee_const:
//  CHECK-NEXT:  result #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  operand #2: {(trivial,fwd,2)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: func_const4:
//  CHECK-NEXT:  result #0: {(trivial4,fwd,0)}
//  CHECK-NEXT: test_tag: func_adj:
//  CHECK-NEXT:  operand #0: {(trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial4,adj,0)}
//  CHECK-NEXT: test_tag: calli4_adj:
//  CHECK-NEXT:  operand #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  operand #2: {(trivial4,adj,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: calli4:
//  CHECK-NEXT:  operand #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  operand #2: {(trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @multi_calli_with_const_arg[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_const @trivial[] {tag = "multi_calli_with_const_arg__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|pp>"}} {tag = "func_pred"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %2 = qwerty.func_const @priv_calli_multi_arg[] {tag = "callee_const"} : () -> !qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>
  %3 = qwerty.call_indirect %2(%arg0, %1) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  %4 = qwerty.func_const @trivial4[] {tag = "func_const4"} : () -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %5 = qwerty.func_adj %4 {tag = "func_adj"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %6 = qwerty.call_indirect %2(%3, %5) {tag = "calli4_adj"} : (!qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  %7 = qwerty.call_indirect %2(%6, %4) {tag = "calli4"} : (!qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "ret"} %7 : !qwerty<qbundle[4]>
}

// -----

// ...But at least it's sound

// CHECK-LABEL: test_tag: priv_calli_multi_arg__func_adj:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: callee_ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func private @priv_calli_multi_arg[](%arg0: !qwerty<qbundle[4]>, %arg1: !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_adj %arg1 {tag = "priv_calli_multi_arg__func_adj"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %1 = qwerty.call_indirect %0(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "callee_ret"} %1 : !qwerty<qbundle[4]>
}

qwerty.func private @trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

qwerty.func private @trivial4[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  qwerty.return %arg0 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: test_tag: multi_calli_and_call_with_const_arg__func_const:
//  CHECK-NEXT:  result #0: {(trivial,fwd,0)}
//  CHECK-NEXT: test_tag: func_pred:
//  CHECK-NEXT:  operand #0: {(trivial,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial,fwd,2)}
//  CHECK-NEXT: test_tag: callee_const:
//  CHECK-NEXT:  result #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  operand #2: {(trivial,fwd,2)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: func_const4:
//  CHECK-NEXT:  result #0: {(trivial4,fwd,0)}
//  CHECK-NEXT: test_tag: func_adj:
//  CHECK-NEXT:  operand #0: {(trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: {(trivial4,adj,0)}
//  CHECK-NEXT: test_tag: calli4_adj:
//  CHECK-NEXT:  operand #0: {(priv_calli_multi_arg,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  operand #2: {(trivial4,adj,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: call4:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  operand #1: {(trivial4,fwd,0)}
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @multi_calli_and_call_with_const_arg[](%arg0: !qwerty<qbundle[4]>) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.func_const @trivial[] {tag = "multi_calli_and_call_with_const_arg__func_const"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|pp>"}} {tag = "func_pred"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %2 = qwerty.func_const @priv_calli_multi_arg[] {tag = "callee_const"} : () -> !qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>
  %3 = qwerty.call_indirect %2(%arg0, %1) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  %4 = qwerty.func_const @trivial4[] {tag = "func_const4"} : () -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %5 = qwerty.func_adj %4 {tag = "func_adj"} : (!qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>
  %6 = qwerty.call_indirect %2(%3, %5) {tag = "calli4_adj"} : (!qwerty<func(!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) irrev-> !qwerty<qbundle[4]>>, !qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  %7 = qwerty.call @priv_calli_multi_arg(%6, %4) {tag = "call4"} : (!qwerty<qbundle[4]>, !qwerty<func(!qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]>>) -> !qwerty<qbundle[4]>
  qwerty.return {tag = "ret"} %7 : !qwerty<qbundle[4]>
}

// -----

// The analysis SparseConstantPropagation needs to be loaded in the solver or
// this test will fail

qwerty.func private @trivial1[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

qwerty.func private @trivial2[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: test_tag: if_call__func_const_then:
//  CHECK-NEXT:  result #0: {(trivial1,fwd,0)}
//  CHECK-NEXT: test_tag: then_yield:
//  CHECK-NEXT:  operand #0: {(trivial1,fwd,0)}
//  CHECK-NEXT: test_tag: func_const_else:
//  CHECK-NEXT:  result #0: {(trivial2,fwd,0)}
//  CHECK-NEXT: test_tag: else_yield:
//  CHECK-NEXT:  operand #0: {(trivial2,fwd,0)}
//  CHECK-NEXT: test_tag: if:
//  CHECK-NEXT:  operand #0: bottom
//  CHECK-NEXT:  result #0: {(trivial1,fwd,0), (trivial2,fwd,0)}
//  CHECK-NEXT: test_tag: calli:
//  CHECK-NEXT:  operand #0: {(trivial1,fwd,0), (trivial2,fwd,0)}
//  CHECK-NEXT:  operand #1: bottom
//  CHECK-NEXT:  result #0: bottom
//  CHECK-NEXT: test_tag: ret:
//  CHECK-NEXT:  operand #0: bottom
qwerty.func @if_call[](%arg0: !qwerty<qbundle[2]>, %arg1: i1) irrev-> !qwerty<qbundle[2]> {
  %0 = scf.if %arg1 -> (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) {
    %1 = qwerty.func_const @trivial1[] {tag = "if_call__func_const_then"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield {tag = "then_yield"} %1 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } else {
    %2 = qwerty.func_const @trivial2[] {tag = "func_const_else"} : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
    scf.yield {tag = "else_yield"} %2 : !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  } {tag = "if"}
  %3 = qwerty.call_indirect %0(%arg0) {tag = "calli"} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return {tag = "ret"} %3 : !qwerty<qbundle[2]>
}
