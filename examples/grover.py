#!/usr/bin/env python3

"""
Implementation of Grover's algorithm as defined in Section 6.1 of Nielsen and
Chuang. Takes three inputs: (1) a classical function `oracle' that returns 1 on
a matching N-bit standard basis state; (2) a number of iterations to run; and
(3) the number of samples to collect (`n_shots').

When run directly, this module also acts as a tester for the Qwerty
implementation of Grover's. Given a number of bits N as input, it searches for
the N-bit standard basis state consisting of all 1s.
"""

import math
from argparse import ArgumentParser
from qwerty import *

def grover(oracle, num_iter, shots=None):
    @qpu[[N]]
    def grover_iter(q):
        return (q | oracle.sign
                  | 'p'**N >> -'p'**N)

    @qpu[[N]]
    def kernel():
        return ('p'**N | (grover_iter for i in range(num_iter))
                       | measure**N)

    results = kernel(shots=shots)
    return list(sorted(x for x in set(results) if oracle(x)))

def calc_num_iter(num_qubits, num_answers):
    n = 2**num_qubits
    m = num_answers
    theta = 2*math.acos(math.sqrt((n-m)/n))
    rnd = lambda x: math.ceil(x-0.5)
    return rnd(math.acos(math.sqrt(m/n))/theta)

def get_black_box(num_qubits):
    @classical
    def all_ones(x: bit[num_qubits]) -> bit:
        return x.and_reduce()

    return all_ones

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_qubits',
                        type=int,
                        help='The number of qubits N on which to run '
                             'fixed-point search.')
    parser.add_argument('--shots', '-s',
                        type=int,
                        default=1024,
                        help='Number of shots. Default: %(default)s')
    args = parser.parse_args()

    num_answers = 1
    num_qubits = args.num_qubits
    oracle = get_black_box(num_qubits)
    num_iter = calc_num_iter(num_qubits, num_answers)
    answers = grover(oracle, num_iter, args.shots)

    for answer in answers:
        print(answer)
