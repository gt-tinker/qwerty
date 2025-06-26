// RUN: qwerty-opt %s --split-input-file --verify-diagnostics

module {
  qwerty.func @sciff_iff_0[]() irrev-> !qwerty<bitbundle[1]> {
    %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %1 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %2 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
    // expected-error@+1 {{Result (0) is not linear with this IR instruction}}
    %4 = scf.if %3 -> (!qwerty<qbundle[1]>) {
      %7 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %7 : !qwerty<qbundle[1]>
    } else {
      %7 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Y[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %7 : !qwerty<qbundle[1]>
    }
    %5 = qwerty.qbmeas %4 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %6 = qwerty.qbmeas %4 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    qwerty.return %5 : !qwerty<bitbundle[1]>
  }
}

// -----

module {
  qwerty.func @no_cloning_0[]() irrev-> !qwerty<bitbundle[1]> {
    %0 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %1 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    qwerty.return %1 : !qwerty<bitbundle[1]>
  }
  qwerty.func @unpack_this_1[]() irrev-> !qwerty<bitbundle[4]> {
    %0 = qwerty.qbprep Z<PLUS>[2] : () -> !qwerty<qbundle[2]>
    %1 = qwerty.qbprep X<PLUS>[2] : () -> !qwerty<qbundle[2]>
    // expected-error@+1 {{Result (0) is not linear with this IR instruction}}
    %2:2 = qwerty.qbunpack %0 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
    %3:2 = qwerty.qbunpack %1 : (!qwerty<qbundle[2]>) -> (!qcirc.qubit, !qcirc.qubit)
    %4 = qwerty.qbpack(%2#0) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
    %5 = qwerty.qbpack(%2#1) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
    %6 = qwerty.qbtrans %4 by {std: Z[1]} >> {std: X[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
    %7 = qwerty.qbtrans %5 by {std: Z[1]} >> {std: X[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
    %8 = qwerty.qbunpack %6 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
    %9 = qwerty.qbunpack %7 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
    %10 = qwerty.qbpack(%3#0) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
    %11 = qwerty.qbpack(%3#1) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
    %12 = qwerty.qbtrans %10 by {std: X[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
    %13 = qwerty.qbtrans %11 by {std: X[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
    %14 = qwerty.qbunpack %12 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
    %15 = qwerty.qbunpack %13 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
    %16 = qwerty.qbpack(%8, %9, %14, %15) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
    %17 = qwerty.qbmeas %16 by {std: Z[4]} : !qwerty<qbundle[4]> -> !qwerty<bitbundle[4]>
    %18 = qwerty.qbpack(%2#0) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
    qwerty.return %17 : !qwerty<bitbundle[4]>
  }
}

// -----

module {
  qwerty.func @sciff_iff_0[]() irrev-> !qwerty<bitbundle[1]> {
    %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
    // expected-error@+1 {{Result (0) is not linear with this IR instruction}}
    %1 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %2 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
    %4 = scf.if %3 -> (!qwerty<qbundle[1]>) {
      %7 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      %8 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %7 : !qwerty<qbundle[1]>
    } else {
      %7 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Y[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %7 : !qwerty<qbundle[1]>
    }
    %5 = qwerty.qbmeas %4 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %6 = qwerty.qbmeas %4 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    qwerty.return %5 : !qwerty<bitbundle[1]>
  }
}

// -----

module {
  qwerty.func @sciff_iff_0[]() irrev-> !qwerty<bitbundle[1]> {
    %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %1 = qwerty.qbprep X<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %2 = qwerty.qbmeas %0 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %3 = qwerty.bitunpack %2 : (!qwerty<bitbundle[1]>) -> i1
    // expected-error@+1 {{Result (0) is not linear with this IR instruction}}
    %4 = scf.if %3 -> (!qwerty<qbundle[1]>) {
      %12 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %12 : !qwerty<qbundle[1]>
    } else {
      %12 = qwerty.qbtrans %1 by {std: X[1]} >> {std: Y[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %12 : !qwerty<qbundle[1]>
    }
    %5 = qwerty.qbprep Z<MINUS>[1] : () -> !qwerty<qbundle[1]>
    %6 = qwerty.qbmeas %5 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %7 = qwerty.qbprep X<MINUS>[1] : () -> !qwerty<qbundle[1]>
    %8 = qwerty.qbmeas %7 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    %9 = qwerty.bitunpack %6 : (!qwerty<bitbundle[1]>) -> i1
    %10 = scf.if %9 -> (!qwerty<qbundle[1]>) {
      %12 = qwerty.qbtrans %4 by {std: Y[1]} >> {std: Z[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
      scf.yield %12 : !qwerty<qbundle[1]>
    } else {
      %12 = qwerty.bitunpack %8 : (!qwerty<bitbundle[1]>) -> i1
      %13 = scf.if %12 -> (!qwerty<qbundle[1]>) {
        %14 = qwerty.qbtrans %4 by {std: Z[1]} >> {std: Y[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
        scf.yield %14 : !qwerty<qbundle[1]>
      } else {
        %14 = qwerty.qbtrans %4 by {std: X[1]} >> {std: Y[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
        %15 = qwerty.qbtrans %4 by {std: X[1]} >> {std: Y[1]} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
        scf.yield %14 : !qwerty<qbundle[1]>
      }
      scf.yield %13 : !qwerty<qbundle[1]>
    }
    %11 = qwerty.qbmeas %10 by {std: Z[1]} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    qwerty.return %11 : !qwerty<bitbundle[1]>
  }
}

