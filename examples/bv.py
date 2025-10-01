#!/usr/bin/env python3

"""
Perform the Bernstein—Vazirani algorithm on a provided bitstring, printing the
bitstring determined by both the quantum algorithm and classical algorithm.
"""

from argparse import ArgumentParser
from qwerty import *

def bv(oracle):
    @qpu[[N]]
    def kernel():
        return ('p'**N | oracle.sign
                       | pm**N >> std**N
                       | measure**N)

    return kernel()

def get_black_box(secret_string):
    @classical[[N]]
    def f(x: bit[N]) -> bit:
        return (secret_string & x).xor_reduce()
    return f

def naive_classical(f, n_bits):
    secret_found = bit[n_bits](0)
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
    args = parser.parse_args()

    secret_str = bit.from_str(args.secret_bits)
    n_bits = len(secret_str)
    black_box = get_black_box(secret_str)

    print('Classical:', naive_classical(black_box, n_bits))
    print('Quantum:  ', bv(black_box))
