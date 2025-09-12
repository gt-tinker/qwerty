#!/usr/bin/env python3

"""
Implementation of quantum phase estimation (QPE) as defined in Section 5.2 of
Nielsen and Chuang. Takes `precision', the number of bits of the phase to find;
a function to prepare the eigenvector whose eigenvalue we aim to find
(`prep_eigvec'); the operator whose eigenvalue to find (`op'); and the number
of shots to run (`n_shots').

When run directly, this module also acts as a tester for the Qwerty
implementation of QPE. The operator whose value to find is a basis translation
that imparts a phase of e^iφ on |1⟩. The angle φ along with the precision of
the estimate are specified on the command line.
"""

from argparse import ArgumentParser
from qwerty import *

def qpe(precision, prep_eigvec, op, n_shots, acc=None):
    @qpu[[M,T]](prep_eigvec, op)
    def kernel(prep_eigvec: qfunc[0,M], op: rev_qfunc[M][[...]]) -> bit[T]:
        return 'p'[T] + prep_eigvec() \
               | (std[T-1-j] + '1' + std[j] & op[[j]] for j in range(T)) \
               | fourier[T].measure + discard[M]

    k_inst = kernel[[precision]]
    for meas in k_inst(shots=n_shots, acc=acc):
        yield meas.as_bin_frac()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('phi',
                        type=float,
                        help='The phase angle (eigenvalue) to estimate. '
                             'Example: 0.12')
    parser.add_argument('precision',
                        type=int,
                        help='The number of bits to estimate. Example: 13')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()
    phi = args.phi
    precision = args.precision

    @qpu
    def prep1() -> qubit:
        return '1'

    @qpu[[J]](phi)
    @reversible
    def rot(phi: angle, q: qubit) -> qubit:
        return q | std >> {'0', '1' @ (360*phi*2**J)}

    print('Expected:', phi)
    phi_got, = qpe(precision, prep1, rot, n_shots=1, acc=args.acc)
    print('Actual:', float(phi_got))
