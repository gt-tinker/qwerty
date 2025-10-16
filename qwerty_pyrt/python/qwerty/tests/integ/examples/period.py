"""
Implementation of period finding as defined in Section 5.4.1 in Nielsen and
Chuang.
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

def test(num_bits, mod, attempts):
    black_box = get_black_box(num_bits, mod)

    for i in range(attempts):
        period_found = period_finding(black_box)
        yield period_found
        if period_found == mod:
            break
