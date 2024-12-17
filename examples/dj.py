#!/usr/bin/env python3

"""
Generalization of deutsch.py. Runs the Deutsch–Jozsa algorithm twice: once on a
"balanced" oracle (black box) and again on a "constant" one.
"""

from argparse import ArgumentParser
from qwerty import *

def deutsch_jozsa(f, acc=None):
    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] | f.sign \
                      | pm[N].measure

    if int(kernel(acc=acc)) == 0:
        return 'constant'
    else:
        return 'balanced'

@classical
def constant(x: bit[4]) -> bit:
    # f(x) = 1
    return bit[1](0b1)

@classical
def balanced(x: bit[4]) -> bit:
    # f(x) = 1 for half the inputs
    # and f(x) = 0 for the other half
    return x.xor_reduce()

def naive_classical(f, n_bits):
    answers = [0, 0]
    for i in range(2**(n_bits-1)+1):
        answer = int(f(bit[n_bits](i)))
        answers[answer] += 1

    if 0 in answers:
        return 'constant'
    else:
        return 'balanced'

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    print('Constant test:')
    print('Classical:', naive_classical(constant, 4))
    print('Quantum:  ', deutsch_jozsa(constant, acc=args.acc))

    print('\nBalanced test:')
    print('Classical:', naive_classical(balanced, 4))
    print('Quantum:  ', deutsch_jozsa(balanced, acc=args.acc))
