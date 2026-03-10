#!/usr/bin/env python3

"""
Implementation of amplitude estimation as defined by Bassard et al. (2000).

When run directly, this module acts as a tester.
"""
# TODO: Explain what it is testing

import math
from argparse import ArgumentParser
from qwerty import *
from qpe import qpe
from grover import get_black_box, get_grover_iter

def ampl_est(prec, num_qubits, shots, acc=None):
    @qpu
    def get_init_state():
        return 'p'**num_qubits

    grover_iter = get_grover_iter(get_black_box(num_qubits))

    @qpu[[J]]
    @reversible
    def op(q):
        return q | (grover_iter for j in range(2**J))

    histo = qpe(prec, get_init_state, op, shots, acc=acc)

    def angle_to_a_estimate(y_angle_frac):
        y_angle_rad = float(y_angle_frac) * 2.0*math.pi
        a_est = math.sin(y_angle_rad)**2
        return a_est

    a_est_histo = {angle_to_a_estimate(y_angle_frac): count
                   for y_angle_frac, count in histo.items()}
    return a_est_histo

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_qubits',
                        type=int,
                        help='The number of qubits N')
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

    a_est_histo = ampl_est(args.precision, args.num_qubits, args.shots,
                           args.acc)

    for a_est, count in a_est_histo.items():
        print('{} -> {:.02f}%'.format(a_est, count/args.shots*100))
