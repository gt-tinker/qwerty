#!/usr/bin/env python3

"""
Implementation of period finding as defined in Section 5.4.1 in Nielsen and
Chuang.

When run directly, this module also acts as a tester for the Qwerty
implementation of period finding. The function whose period is found is f(x) =
x % M. The number of input/output bits for the black box and the modulus M are
passed on the command line. The result from the algorithm is then checked for
correctness.
"""

import math
from fractions import Fraction
from argparse import ArgumentParser
from qwerty import *

def period_finding(f):
    @qpu[[N]]
    def kernel():
        return ('p'**N * '0'**N
                | f.xor
                | id**N * discard**N
                | fourier[N].measure)

    def shift_binary_point(bits):
        return Fraction(int(bits),
                        2**len(bits))

    frac1 = shift_binary_point(kernel())
    frac2 = shift_binary_point(kernel())
    return math.lcm(frac1.denominator,
                    frac2.denominator)

def get_black_box(n_bits, modulus):
    @classical
    def mod(x: bit[n_bits]) -> bit[n_bits]:
        return x % modulus

    return mod

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_bits',
                        type=int,
                        help='Number of input bits to the black box')
    parser.add_argument('--mod', '-m',
                        type=int,
                        metavar='M',
                        help='Find the period of f(x) = x % M. Must be a '
                             'power of 2. Default: 2**(N-1)')
    args = parser.parse_args()

    if args.num_bits < 3:
        raise ValueError('number of bits must be at least 3')

    if args.mod is None:
        mod = 2**(args.num_bits-1)
    else:
        if args.mod.bit_count() != 1:
            raise ValueError(f'modulus {args.mod} is not a power of 2')
        if args.mod.bit_length() > args.num_bits:
            raise ValueError(f'modulus {args.mod} is too big for number of '
                             f'bits {args.num_bits}')
        mod = args.mod

    print(f'Finding period of f(x) = x % {mod}...')
    black_box = get_black_box(args.num_bits, mod)

    for i in range(16):
        period_found = period_finding(black_box)
        if period_found == mod:
            print(f'Found period: {period_found}')
            break
        else:
            print('Period finding failed. Trying again...')
    else:
        print('Exceeded number of tries for period finding')
