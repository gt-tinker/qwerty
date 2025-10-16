// RUN: qwerty-opt %s | qwerty-opt | FileCheck --strict-whitespace %s
// Use strict whitespace because whitespace honestly makes a difference in
// readability, and this file's job is to test printers/parsers

// CHECK-LABEL: qwerty.func @phase[%arg0: f64](%arg1: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbphase %arg1 by exp(i*%arg0) : (f64, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @phase[%cap0: f64](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.qbphase %arg0 by exp(i*%cap0) : (f64, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @no_captures[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbid %arg0 : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @no_captures[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.qbid %arg0 : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @not_rev[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<bitbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbmeas %arg0 by {std: Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<bitbundle[2]>
//  CHECK-NEXT: }
qwerty.func @not_rev[](%arg0: !qwerty<qbundle[2]>) irrev-> !qwerty<bitbundle[2]> {
  %0 = qwerty.qbmeas %arg0 by {std: Z[2]} : !qwerty<qbundle[2]> -> !qwerty<bitbundle[2]>
  qwerty.return %0 : !qwerty<bitbundle[2]>
}

// CHECK-LABEL: qwerty.func @no_args[%arg0: !qwerty<qbundle[2]>]() irrev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbproj %arg0 by {std: X[2]} : !qwerty<qbundle[2]> -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @no_args[%cap0: !qwerty<qbundle[2]>]() irrev-> !qwerty<qbundle[2]> {
  %0 = qwerty.qbproj %cap0 by {std: X[2]} : !qwerty<qbundle[2]> -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func private @private_func[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.qbtrans %arg0 by {std: Z[2]} >> {std: X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func private @private_func[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.qbtrans %arg0 by {std: Z[2]} >> {std: X[2]} : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @call_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.call adj @private_func(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @call_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.call adj @private_func(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @call_not_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.call @private_func(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @call_not_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.call @private_func(%arg0) : (!qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %0 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @func_const[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @private_func[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @func_const[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @private_func[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.call_indirect %0(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %1 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @func_const_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
//  CHECK-NEXT:   %0 = qwerty.func_const @private_func[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
//  CHECK-NEXT:   %2 = qwerty.call_indirect %1(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
//  CHECK-NEXT:   qwerty.return %2 : !qwerty<qbundle[2]>
//  CHECK-NEXT: }
qwerty.func @func_const_adj[](%arg0: !qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]> {
  %0 = qwerty.func_const @private_func[] : () -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %1 = qwerty.func_adj %0 : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>) -> !qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>
  %2 = qwerty.call_indirect %1(%arg0) : (!qwerty<func(!qwerty<qbundle[2]>) rev-> !qwerty<qbundle[2]>>, !qwerty<qbundle[2]>) -> !qwerty<qbundle[2]>
  qwerty.return %2 : !qwerty<qbundle[2]>
}

// CHECK-LABEL: qwerty.func @qbinit[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbinit %arg0 as {list:{"|pm>"}, list:{"|ji>"}} : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %1 = qwerty.qbdeinit %0 as {list:{"|pm>"}, list:{"|ji>"}} : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
qwerty.func @qbinit[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
  %0 = qwerty.qbinit %arg0 as {list:{"|pm>"}, list:{"|ji>"}} : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  %1 = qwerty.qbdeinit %0 as {list:{"|pm>"}, list:{"|ji>"}} : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %1 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: qwerty.func @qbinit_phases[](%arg0: !qwerty<qbundle[4]>, %arg1: f64) irrev-> !qwerty<qbundle[4]> {
//  CHECK-NEXT:   %0 = qwerty.qbinit %arg0 as {list:{"|pm>"}, list:{exp(i*theta)*"|ji>"}} phases (%arg1) : (f64, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   %1 = qwerty.qbdeinit %0 as {list:{"|pm>"}, list:{exp(i*theta)*"|ji>"}} phases (%arg1) : (f64, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
//  CHECK-NEXT:   qwerty.return %1 : !qwerty<qbundle[4]>
//  CHECK-NEXT: }
qwerty.func @qbinit_phases[](%arg0: !qwerty<qbundle[4]>, %arg1: f64) irrev-> !qwerty<qbundle[4]> {
  %0 = qwerty.qbinit %arg0 as {list:{"|pm>"}, list:{exp(i*theta)*"|ji>"}} phases (%arg1) : (f64, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  %1 = qwerty.qbdeinit %0 as {list:{"|pm>"}, list:{exp(i*theta)*"|ji>"}} phases (%arg1) : (f64, !qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
  qwerty.return %1 : !qwerty<qbundle[4]>
}

// CHECK-LABEL: qwerty.func @pred_inlined[](%arg0: !qwerty<qbundle[5]>, %arg1: !qwerty<qbundle[4]>) irrev-> (!qwerty<qbundle[5]>, !qwerty<qbundle[4]>) {
//  CHECK-NEXT:   %predBundleOut, %regionResult = qwerty.pred on {list:{"|111>"}, std: X[2]} (%arg0: !qwerty<qbundle[5]>, %arg1 as %arg2: !qwerty<qbundle[4]>) -> (!qwerty<qbundle[5]>, !qwerty<qbundle[4]>) {
//  CHECK-NEXT:      qwerty.yield %arg2 : !qwerty<qbundle[4]>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   qwerty.return %predBundleOut, %regionResult : !qwerty<qbundle[5]>, !qwerty<qbundle[4]>
//  CHECK-NEXT: }
qwerty.func @pred_inlined[](%arg0: !qwerty<qbundle[5]>, %arg1: !qwerty<qbundle[4]>) irrev-> (!qwerty<qbundle[5]>, !qwerty<qbundle[4]>) {
  %predBundleOut, %regionResult = qwerty.pred on {list:{"|111>"}, std: X[2]} (%arg0: !qwerty<qbundle[5]>, %arg1 as %arg2: !qwerty<qbundle[4]>) -> (!qwerty<qbundle[5]>, !qwerty<qbundle[4]>) {
     qwerty.yield %arg2 : !qwerty<qbundle[4]>
  }
  qwerty.return %predBundleOut, %regionResult : !qwerty<qbundle[5]>, !qwerty<qbundle[4]>
}

// CHECK-LABEL: qwerty.func @revolve[](%arg0: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
//  CHECK-NEXT:   %0 = qwerty.qbtrans %arg0 by {std: Z[5]} >> {revolve: {std: X[4]} by {"|0>", "|1>"}} : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
//  CHECK-NEXT:   qwerty.return %0 : !qwerty<qbundle[5]>
//  CHECK-NEXT: }
qwerty.func @revolve[](%arg0: !qwerty<qbundle[5]>) irrev-> !qwerty<qbundle[5]> {
  %0 = qwerty.qbtrans %arg0 by {std: Z[5]} >> {revolve: {std: X[4]} by {"|0>", "|1>"}} : (!qwerty<qbundle[5]>) -> !qwerty<qbundle[5]>
  qwerty.return %0 : !qwerty<qbundle[5]>
}
