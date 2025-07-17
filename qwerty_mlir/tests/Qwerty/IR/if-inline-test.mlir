// RUN: qwerty-opt -split-input-file -canonicalize %s | FileCheck %s

qwerty.func private @ecsg_0__lambda0[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbmeas %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  qwerty.return %0 : !qwerty<bitbundle[1]>
}
qwerty.func private @ecsg_0__lambda1[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: X[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
  qwerty.func private @ecsg_0__lambda2[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @ecsg_0__lambda3[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @ecsg_0__lambda4[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbmeas %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  qwerty.return %0 : !qwerty<bitbundle[1]>
}

// CHECK-LABEL: qwerty.func @ecsg_0[]() irrev-> !qwerty<bitbundle[1]> {
//  CHECK-NEXT:    %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %1 = qwerty.call @ecsg_0__lambda0(%0) : (!qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:    %2 = qwerty.bitunpack %1 : (!qwerty<bitbundle[1]>) -> i1
//  CHECK-NEXT:    %3 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %4 = scf.if %2 -> (!qwerty<qbundle[1]>) {
//  CHECK-NEXT:      %8 = qwerty.call @ecsg_0__lambda1(%3) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %8 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    } else {
//  CHECK-NEXT:      %8 = qwerty.call @ecsg_0__lambda2(%3) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %8 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %5 = qwerty.call @ecsg_0__lambda3(%4) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %6 = scf.if %2 -> (!qwerty<qbundle[1]>) {
//  CHECK-NEXT:      %8 = qwerty.call @ecsg_0__lambda1(%5) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %8 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    } else {
//  CHECK-NEXT:      %8 = qwerty.call @ecsg_0__lambda2(%5) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %8 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %7 = qwerty.call @ecsg_0__lambda4(%6) : (!qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:    qwerty.return %7 : !qwerty<bitbundle[1]>
//  CHECK-NEXT:  }
qwerty.func @ecsg_0[]() irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.func_const @ecsg_0__lambda0[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>
  %2 = qwerty.call_indirect %1(%0) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
  %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
  %4 = scf.if %3 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
      %12 = qwerty.func_const @ecsg_0__lambda1[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %12 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  } else {
      %12 = qwerty.func_const @ecsg_0__lambda2[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %12 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  }
  %5 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %6 = qwerty.call_indirect %4(%5) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %7 = qwerty.func_const @ecsg_0__lambda3[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  %8 = qwerty.call_indirect %7(%6) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %9 = qwerty.call_indirect %4(%8) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %10 = qwerty.func_const @ecsg_0__lambda4[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>
  %11 = qwerty.call_indirect %10(%9) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
  qwerty.return %11 : !qwerty<bitbundle[1]>
}

// -----

qwerty.func private @decsg_0__lambda0[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbmeas %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  qwerty.return %0 : !qwerty<bitbundle[1]>
}
qwerty.func private @decsg_0__lambda1[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: X[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @decsg_0__lambda2[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
  qwerty.func private @decsg_0__lambda3[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Y[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @decsg_0__lambda4[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @decsg_0__lambda5[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbmeas %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  qwerty.return %0 : !qwerty<bitbundle[1]>
}

// CHECK-LABEL: qwerty.func @decsg_0[]() irrev-> !qwerty<bitbundle[1]> {
//  CHECK-NEXT:    %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %1 = qwerty.call @decsg_0__lambda0(%0) : (!qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:    %2 = qwerty.bitunpack %1 : (!qwerty<bitbundle[1]>) -> i1
//  CHECK-NEXT:    %3 = qwerty.bitunpack %1 : (!qwerty<bitbundle[1]>) -> i1
//  CHECK-NEXT:    %4 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %5 = scf.if %2 -> (!qwerty<qbundle[1]>) {
//  CHECK-NEXT:     %10 = qwerty.call @decsg_0__lambda1(%4) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    } else {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda2(%4) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %6 = scf.if %3 -> (!qwerty<qbundle[1]>) {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda3(%5) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    } else {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda4(%5) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %7 = scf.if %2 -> (!qwerty<qbundle[1]>) {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda1(%6) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    } else {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda2(%6) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %8 = scf.if %3 -> (!qwerty<qbundle[1]>) {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda3(%7) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:      %10 = qwerty.call @decsg_0__lambda4(%7) : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:      scf.yield %10 : !qwerty<qbundle[1]>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %9 = qwerty.call @decsg_0__lambda5(%8) : (!qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:    qwerty.return %9 : !qwerty<bitbundle[1]>
//  CHECK-NEXT:  }
qwerty.func @decsg_0[]() irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.func_const @decsg_0__lambda0[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>
  %2 = qwerty.call_indirect %1(%0) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
  %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
  %4 = scf.if %3 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
      %14 = qwerty.func_const @decsg_0__lambda1[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %14 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  } else {
      %14 = qwerty.func_const @decsg_0__lambda2[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %14 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  }
  %5 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
  %6 = scf.if %5 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
      %14 = qwerty.func_const @decsg_0__lambda3[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %14 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  } else {
      %14 = qwerty.func_const @decsg_0__lambda4[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %14 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  }
  %7 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %8 = qwerty.call_indirect %4(%7) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %9 = qwerty.call_indirect %6(%8) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %10 = qwerty.call_indirect %4(%9) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %11 = qwerty.call_indirect %6(%10) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %12 = qwerty.func_const @decsg_0__lambda5[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>
  %13 = qwerty.call_indirect %12(%11) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<bitbundle[1]>
  qwerty.return %13 : !qwerty<bitbundle[1]>
}

// -----

qwerty.func private @fail_0__lambda2[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: X[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @fail_0__lambda3[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @fail_0__lambda5[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Y[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}
qwerty.func private @fail_0__lambda6[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %0 = qwerty.qbproj %arg0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}

// CHECK-LABEL: qwerty.func @fail_0[]() irrev-> !qwerty<bitbundle[1]> {
//  CHECK-NEXT:    %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %1 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %2 = qwerty.qbmeas %1 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:    %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
//  CHECK-NEXT:    %4 = scf.if %3 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
//  CHECK-NEXT:      %10 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:      %11 = qwerty.bitunpack %10 : (!qwerty<bitbundle[1]>) -> i1
//  CHECK-NEXT:      %12 = scf.if %11 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
//  CHECK-NEXT:        %13 = qwerty.func_const @fail_0__lambda2[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:        scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:      } else {
//  CHECK-NEXT:        %13 = qwerty.func_const @fail_0__lambda3[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:        scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:      }
//  CHECK-NEXT:      scf.yield %12 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:    } else {
//  CHECK-NEXT:      %10 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:      %11 = qwerty.bitunpack %10 : (!qwerty<bitbundle[1]>) -> i1
//  CHECK-NEXT:      %12 = scf.if %11 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
//  CHECK-NEXT:        %13 = qwerty.func_const @fail_0__lambda5[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:        scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:      } else {
//  CHECK-NEXT:        %13 = qwerty.func_const @fail_0__lambda6[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:        scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:      }
//  CHECK-NEXT:      scf.yield %12 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
//  CHECK-NEXT:    }
//  CHECK-NEXT:    %5 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %6 = qwerty.call_indirect %4(%5) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %7 = qwerty.qbproj %6 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %8 = qwerty.call_indirect %4(%7) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
//  CHECK-NEXT:    %9 = qwerty.qbmeas %8 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
//  CHECK-NEXT:    qwerty.return %9 : !qwerty<bitbundle[1]>
//  CHECK-NEXT:  }
qwerty.func @fail_0[]() irrev-> !qwerty<bitbundle[1]> {
  %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %1 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %2 = qwerty.qbmeas %1 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
  %4 = scf.if %3 -> !qwerty<bitbundle[1]> {
      %10 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
      scf.yield %10 : !qwerty<bitbundle[1]>
  } else {
      %10 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
      scf.yield %10 : !qwerty<bitbundle[1]>
  }
  %5 = scf.if %3 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
      %11 = qwerty.bitunpack %4 : (!qwerty<bitbundle[1]>) -> i1
      %12 = scf.if %11 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
      %13 = qwerty.func_const @fail_0__lambda2[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      } else {
      %13 = qwerty.func_const @fail_0__lambda3[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      }
      scf.yield %12 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  } else {
      %11 = qwerty.bitunpack %4 : (!qwerty<bitbundle[1]>) -> i1
      %12 = scf.if %11 -> (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>) {
      %13 = qwerty.func_const @fail_0__lambda5[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      } else {
      %13 = qwerty.func_const @fail_0__lambda6[] : () -> !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      scf.yield %13 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
      }
      scf.yield %12 : !qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>
  }
  %6 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
  %7 = qwerty.call_indirect %5(%6) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %8 = qwerty.qbproj %7 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<qbundle[1]>
  %9 = qwerty.call_indirect %5(%8) : (!qwerty<func(!qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]>>, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  %10 = qwerty.qbmeas %9 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
  qwerty.return %10 : !qwerty<bitbundle[1]>
}
