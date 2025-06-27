module {
  qwerty.func private @f_0__xor[](%arg0: !qwerty<qbundle[4]>) rev-> !qwerty<qbundle[4]> {
    %0:4 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
    %controlResults, %result = qcirc.gate1q[%0#0]:X %0#3 : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %controlResults_0, %result_1 = qcirc.gate1q[%0#1]:X %result : (!qcirc.qubit, !qcirc.qubit) -> (!qcirc.qubit, !qcirc.qubit)
    %1 = qwerty.qbpack(%controlResults, %controlResults_0, %0#2, %result_1) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
    qwerty.return %1 : !qwerty<qbundle[4]>
  }
  qwerty.func private @kernel_1__lambda0[](%arg0: !qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]> {
    %0 = qwerty.qbtrans %arg0 by {std: X[3]} >> {std: Z[3]} : (!qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
    qwerty.return %0 : !qwerty<qbundle[3]>
  }
  qwerty.func private @kernel_1__lambda1[](%arg0: !qwerty<qbundle[3]>) irrev-> !qwerty<bitbundle[3]> {
    %0 = qwerty.qbmeas %arg0 by {std: Z[3]} : !qwerty<qbundle[3]> -> !qwerty<bitbundle[3]>
    qwerty.return %0 : !qwerty<bitbundle[3]>
  }
  qwerty.func @kernel_1[]() irrev-> !qwerty<bitbundle[3]> {
    %0 = qwerty.qbprep X<PLUS>[3] : () -> !qwerty<qbundle[3]>
    %1 = qwerty.func_const @f_0__sign[] : () -> !qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>
    %2 = qwerty.call_indirect %1(%0) : (!qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>, !qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
    %3 = qwerty.func_const @kernel_1__lambda0[] : () -> !qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>
    %4 = qwerty.call_indirect %3(%2) : (!qwerty<func(!qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]>>, !qwerty<qbundle[3]>) -> !qwerty<qbundle[3]>
    %5 = qwerty.func_const @kernel_1__lambda1[] : () -> !qwerty<func(!qwerty<qbundle[3]>) irrev-> !qwerty<bitbundle[3]>>
    %6 = qwerty.call_indirect %5(%4) : (!qwerty<func(!qwerty<qbundle[3]>) irrev-> !qwerty<bitbundle[3]>>, !qwerty<qbundle[3]>) -> !qwerty<bitbundle[3]>
    qwerty.return %6 : !qwerty<bitbundle[3]>
  }
  qwerty.func private @f_0__sign[](%arg0: !qwerty<qbundle[3]>) rev-> !qwerty<qbundle[3]> {
    %0 = qwerty.qbprep Z<PLUS>[1] : () -> !qwerty<qbundle[1]>
    %1 = qwerty.qbinit %0 as {list:{"|m>"}} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
    %2 = qwerty.qbunpack %1 : (!qwerty<qbundle[1]>) -> !qcirc.qubit
    %3:3 = qwerty.qbunpack %arg0 : (!qwerty<qbundle[3]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
    %4 = qwerty.qbpack(%3#0, %3#1, %3#2, %2) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[4]>
    %5 = qwerty.call @f_0__xor(%4) : (!qwerty<qbundle[4]>) -> !qwerty<qbundle[4]>
    %6:4 = qwerty.qbunpack %5 : (!qwerty<qbundle[4]>) -> (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit, !qcirc.qubit)
    %7 = qwerty.qbpack(%6#0, %6#1, %6#2) : (!qcirc.qubit, !qcirc.qubit, !qcirc.qubit) -> !qwerty<qbundle[3]>
    %8 = qwerty.qbpack(%6#3) : (!qcirc.qubit) -> !qwerty<qbundle[1]>
    %9 = qwerty.qbdeinit %8 as {list:{"|m>"}} : (!qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
    qwerty.qbdiscardz %9 : (!qwerty<qbundle[1]>) -> ()
    qwerty.return %7 : !qwerty<qbundle[3]>
  }
}

