// RUN: qwerty-opt -split-input-file -convert-qwerty-to-qcirc -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func private @direct_call__trivial(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
qwerty.func private @direct_call__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @direct_call(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   %0 = call @direct_call__trivial(%arg0) : (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT:   return %0 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
qwerty.func @direct_call[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %1 = qwerty.call @direct_call__trivial(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// -----

// CHECK-LABEL: func.func private @direct_adj_call__trivial__adj(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
qwerty.func private @direct_adj_call__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @direct_adj_call(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   %0 = call @direct_adj_call__trivial__adj(%arg0) : (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT:   return %0 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
qwerty.func @direct_adj_call[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %1 = qwerty.call adj @direct_adj_call__trivial(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// -----

// CHECK-LABEL: func.func private @direct_pred_call__trivial__fwd__ctrl3(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func private @direct_pred_call__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @direct_pred_call(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = call @direct_pred_call__trivial__fwd__ctrl3(%arg0) : (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @direct_pred_call[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %1 = qwerty.call pred {list:{"|1>"}, list:{"|11>"}} @direct_pred_call__trivial(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %1 : !qwerty<qbundle[5]>
}

// -----

// CHECK-LABEL: func.func private @direct_adj_pred_call__trivial__adj__ctrl3(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func private @direct_adj_pred_call__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @direct_adj_pred_call(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = call @direct_adj_pred_call__trivial__adj__ctrl3(%arg0) : (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @direct_adj_pred_call[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %1 = qwerty.call adj pred {list:{"|1>"}, list:{"|11>"}} @direct_adj_pred_call__trivial(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %1 : !qwerty<qbundle[5]>
}

// CHECK-LABEL: func.func private @const_calli__trivial(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qcirc.callable_metadata private @const_calli__trivial__metadata captures [] specs [(fwd,0,@const_calli__trivial,(!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>)]
qwerty.func private @const_calli__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @const_calli(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %1 = qcirc.callable_invoke %0(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>, !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT:   return %1 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
qwerty.func @const_calli[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @const_calli__trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// -----

// CHECK-LABEL: func.func private @const_adj_calli__trivial__adj(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qcirc.callable_metadata private @const_adj_calli__trivial__metadata captures [] specs [(adj,0,@const_adj_calli__trivial__adj,(!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>)]
qwerty.func private @const_adj_calli__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @const_adj_calli(%arg0: !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_adj_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %1 = qcirc.callable_adj %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %2 = qcirc.callable_invoke %1(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>, !qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT:   return %2 : !qcirc<array<!qcirc.qubit>[2]>
//  CHECK-NEXT: }
qwerty.func @const_adj_calli[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @const_adj_calli__trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.call_indirect %1(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %2 : !qwerty<qbundle[2]>
}

// -----

// CHECK-LABEL: func.func private @const_pred_calli__trivial__fwd__ctrl3(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qcirc.callable_metadata private @const_pred_calli__trivial__metadata captures [] specs [(fwd,3,@const_pred_calli__trivial__fwd__ctrl3,(!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>)]
qwerty.func private @const_pred_calli__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @const_pred_calli(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_pred_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %1 = qcirc.callable_ctrl %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %2 = qcirc.callable_invoke %1(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %2 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @const_pred_calli[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.func_const @const_pred_calli__trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|1>"}, list:{"|11>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %2 = qwerty.call_indirect %1(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %2 : !qwerty<qbundle[5]>
}

// -----

// CHECK-LABEL: func.func private @const_adj_pred_calli__trivial__adj__ctrl3(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qcirc.callable_metadata private @const_adj_pred_calli__trivial__metadata captures [] specs [(adj,3,@const_adj_pred_calli__trivial__adj__ctrl3,(!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>)]
qwerty.func private @const_adj_pred_calli__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: func.func @const_adj_pred_calli(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_adj_pred_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %1 = qcirc.callable_adj %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %2 = qcirc.callable_ctrl %1 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %3 = qcirc.callable_invoke %2(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %3 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @const_adj_pred_calli[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.func_const @const_adj_pred_calli__trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.func_pred %1 by {list:{"|1>"}, list:{"|11>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %3 = qwerty.call_indirect %2(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %3 : !qwerty<qbundle[5]>
}

// -----

// CHECK-LABEL: func.func private @const_pred_adj_calli__trivial__adj__ctrl3(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   return %arg0 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
//  CHECK-NEXT: qcirc.callable_metadata private @const_pred_adj_calli__trivial__metadata captures [] specs [(adj,3,@const_pred_adj_calli__trivial__adj__ctrl3,(!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>)]
qwerty.func private @const_pred_adj_calli__trivial[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

//  CHECK-NEXT: func.func @const_pred_adj_calli(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_pred_adj_calli__trivial__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %1 = qcirc.callable_ctrl %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %2 = qcirc.callable_adj %1 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %3 = qcirc.callable_invoke %2(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %3 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @const_pred_adj_calli[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.func_const @const_pred_adj_calli__trivial[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_pred %0 by {list:{"|1>"}, list:{"|11>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %2 = qwerty.func_adj %1 : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %3 = qwerty.call_indirect %2(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %3 : !qwerty<qbundle[5]>
}

// -----

// CHECK-LABEL: func.func @func_pred_already_ones(%arg0: !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>> {
//  CHECK-NEXT:   %0 = qcirc.callable_ctrl %arg0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   return %0 : !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT: }
qwerty.func @func_pred_already_ones[](%arg0: !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) irrev-> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>> {
  %0 = qwerty.func_pred %arg0 by {list:{"|11>"}, list:{"|1>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  qwerty.return %0 : !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
}

// -----

qwerty.func private @direct_call[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  qwerty.return %arg0 : !qwerty<qbundle[5]>
}

qwerty.func private @const_calli[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  qwerty.return %arg0 : !qwerty<qbundle[5]>
}

qwerty.func private @direct_pred_call[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

qwerty.func private @const_pred_calli[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  qwerty.return %arg0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: func.func @multi_calls(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = call @direct_call(%arg0) : (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   %1 = qcirc.callable_create @const_calli__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %2 = qcirc.callable_invoke %1(%0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %2 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @multi_calls[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.call @direct_call(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %1 = qwerty.func_const @const_calli[] : () -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %2 = qwerty.call_indirect %1(%0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %2 : !qwerty<qbundle[5]>
}

// CHECK-LABEL: func.func @multi_adj_calls(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_calli__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %1 = qcirc.callable_adj %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %2 = qcirc.callable_invoke %1(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   %3 = call @direct_call__adj(%2) : (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %3 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @multi_adj_calls[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.func_const @const_calli[] : () -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %2 = qwerty.call_indirect %1(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %3 = qwerty.call adj @direct_call(%2) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %3 : !qwerty<qbundle[5]>
}

// CHECK-LABEL: func.func @multi_calls_pred(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = call @direct_pred_call__fwd__ctrl3(%arg0) : (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   %1 = qcirc.callable_create @const_pred_calli__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %2 = qcirc.callable_ctrl %1 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %3 = qcirc.callable_invoke %2(%0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %3 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @multi_calls_pred[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.call pred {list:{"|1>"}, list:{"|11>"}} @direct_pred_call(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %1 = qwerty.func_const @const_pred_calli[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.func_pred %1 by {list:{"|1>"}, list:{"|11>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %3 = qwerty.call_indirect %2(%0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %3 : !qwerty<qbundle[5]>
}

// CHECK-LABEL: func.func @multi_adj_calls_pred(%arg0: !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]> {
//  CHECK-NEXT:   %0 = qcirc.callable_create @const_pred_calli__metadata[] : () -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %1 = qcirc.callable_adj %0 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>
//  CHECK-NEXT:   %2 = qcirc.callable_ctrl %1 : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[2]>) -> !qcirc<array<!qcirc.qubit>[2]>>) -> !qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>
//  CHECK-NEXT:   %3 = qcirc.callable_invoke %2(%arg0) : (!qcirc<callable (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>>, !qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   %4 = call @direct_pred_call__adj__ctrl3(%arg0) : (!qcirc<array<!qcirc.qubit>[5]>) -> !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT:   return %4 : !qcirc<array<!qcirc.qubit>[5]>
//  CHECK-NEXT: }
qwerty.func @multi_adj_calls_pred[](%arg0: !qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]> {
  %0 = qwerty.func_const @const_pred_calli[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.func_pred %1 by {list:{"|1>"}, list:{"|11>"}} : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>
  %3 = qwerty.call_indirect %2(%arg0) : (!qwerty<func(!qwerty<qbundle[5]>) rev-> !qwerty<qbundle[5]>>, !qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  %4 = qwerty.call adj pred {list:{"|1>"}, list:{"|11>"}} @direct_pred_call(%arg0) : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %4 : !qwerty<qbundle[5]>
}
