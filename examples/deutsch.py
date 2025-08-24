#!/usr/bin/env python3

"""
Runs Deutsch's algorithm twice: once on a "balanced" oracle (black box) and
again on a "constant" one.
"""

from argparse import ArgumentParser
from qwerty import *

def deutsch(f, acc=None):
    @qpu(f)
    def kernel(f: cfunc) -> bit:
        return 'p' | f.sign | pm.measure

    return kernel(acc=acc)

@classical
def balanced(x: bit) -> bit:
    return ~x

@classical
def constant(x: bit) -> bit:
    return bit[1](0b1)

def naive_classical(f):
    return f(bit[1](0b0)) ^ f(bit[1](0b1))

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    print('Balanced f:')
    print('Classical: f(0) xor f(1) =', naive_classical(balanced))
    print('Quantum:   f(0) xor f(1) =', deutsch(balanced, acc=args.acc))

    print('\nConstant f:')
    print('Classical: f(0) xor f(1) =', naive_classical(constant))
    print('Quantum:   f(0) xor f(1) =', deutsch(constant, acc=args.acc))
