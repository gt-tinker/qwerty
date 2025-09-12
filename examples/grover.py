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

def grover(oracle, n_iter, n_shots, acc=None):
    @qpu[[N]](oracle)
    def grover_iter(oracle: cfunc[N,1], q: qubit[N]) -> qubit[N]:
        return q | oracle.sign \
                 | -('p'[N] >> -'p'[N])

    @qpu[[N,I]](grover_iter)
    def kernel(grover_iter: qfunc[N]) -> bit[N]:
        return 'p'[N] | (grover_iter for _ in range(I)) \
                      | std[N].measure

    kern_inst = kernel[[n_iter]]
    results = kern_inst(shots=n_shots, acc=acc)
    return {r for r in set(results) if oracle(r)}

def get_n_iter(n_qubits, n_answers):
    n = 2**n_qubits
    m = n_answers
    theta = 2*math.acos(math.sqrt((n-m)/n))
    rnd = lambda x: math.ceil(x-0.5)
    return rnd(math.acos(math.sqrt(m/n))/theta)

@classical[[N]]
def all_ones(x: bit[N]) -> bit:
    return x.and_reduce()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_qubits',
                        type=int,
                        help='The number of qubits N on which to run '
                             'fixed-point search.')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    n_ans = 1
    n_qubits = args.num_qubits
    oracle = all_ones[[n_qubits]]
    n_iter = get_n_iter(n_qubits, n_ans)
    answers = grover(oracle, n_iter, n_shots=32, acc=args.acc)

    for answer in answers:
        print(answer)
