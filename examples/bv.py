#!/usr/bin/env python3

"""
Perform the Bernstein—Vazirani algorithm on a provided bitstring, printing the
bitstring determined by both the quantum algorithm and classical algorithm.
"""

from argparse import ArgumentParser
from qwerty import *

def bv(f, acc=None):
    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] | f.sign \
                      | pm[N] >> std[N] \
                      | std[N].measure
    return kernel(acc=acc)

def get_black_box(secret_string):
    @classical[[N]](secret_string)
    def f(secret_string: bit[N], x: bit[N]) -> bit:
        return (secret_string & x).xor_reduce()
    return f

def naive_classical(f, n_bits):
    secret_found = bit[n_bits](0b0)
    x = bit[n_bits](0b1 << (n_bits-1))
    for i in range(n_bits):
        secret_found[i] = f(x)
        x = x >> 1
    return secret_found

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('secret_bits',
                        help='The secret bitstring used to define the black '
                             'box for f(x) (i.e., the oracle). Example: 1101')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    secret_str = bit.from_str(args.secret_bits)
    n_bits = len(secret_str)
    black_box = get_black_box(secret_str)

    print('Classical:', naive_classical(black_box, n_bits))
    print('Quantum:  ', bv(black_box, acc=args.acc))
