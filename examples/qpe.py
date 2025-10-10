#!/usr/bin/env python3

"""
Implementation of quantum phase estimation (QPE) as defined in Section 5.2 of
Nielsen and Chuang. Takes `prec`[ision], the number of bits of the phase to find;
a function to prepare the eigenvector whose eigenvalue we aim to find
(`get_init_state'); the operator whose eigenvalue to find (`op'); and the number
of shots to run (`shots').

When run directly, this module also acts as a tester for the Qwerty
implementation of QPE. The operator whose value to find is a basis translation
that tilts the '1' state by φ degrees. The angle φ along with the precision of
the estimate are specified on the command line.
"""

from fractions import Fraction
from argparse import ArgumentParser
from qwerty import *

def qpe(prec, get_init_state, op, shots, acc=None):
    @qpu[[M]]
    def kernel():
        return ('p'**prec * get_init_state()
                | (op[[prec-1-j]]
                   in '?'**j * '1' * '?'**(prec-1-j) * '_'**M
                   for j in range(prec))
                | fourier[prec].measure
                  * discard**M)

    def bits_to_angle_frac(bits):
        return Fraction(int(bits),
                        2**len(bits))

    bits_histo = kernel(shots=shots, acc=acc)
    angle_histo = {bits_to_angle_frac(bits): count
                   for bits, count in bits_histo.items()}
    return angle_histo

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('angle_deg',
                        nargs='?',
                        default=225.0,
                        type=float,
                        help='The tilt angle (eigenvalue) to estimate. '
                             'Default: %(default)s degrees')
    parser.add_argument('precision',
                        nargs='?',
                        default=3,
                        type=int,
                        help='The number of bits of the angle to estimate. '
                             'Default: %(default)s bits')
    parser.add_argument('--shots', '-s',
                        type=int,
                        default=1024,
                        help='Number of shots. Default: %(default)s')
    parser.add_argument('--acc', '-a',
                        default=None,
                        help='Name of an accelerator. The default is local '
                             'simulation.')
    args = parser.parse_args()
    angle_deg, precision, shots = args.angle_deg, args.precision, args.shots

    @qpu
    def init1():
        return '1'

    @qpu[[J]]
    @reversible
    def tilt_op(q):
        return q | '1' >> '1'@(angle_deg*2**J)

    print('Expected:', angle_deg)
    print('Actual:')
    angle_histo = qpe(precision, init1, tilt_op, shots, acc=args.acc)
    for angle_frac, count in angle_histo.items():
        print('{}° -> {:.02f}%'.format(float(360*angle_frac), count/shots*100))
