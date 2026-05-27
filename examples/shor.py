#!/usr/bin/env python3

"""
Shor's integer factoring algorithm as defined in 5.3.2 of Nielsen and Chuang.
Takes an error allowance and integer as input and returns a nontrivial factor.

When run directly, this module also acts as a tester for the Qwerty
implementation of Shor's. The desired integer is passed on the command line and
a nontrivial factor (or an error) is returned.
"""

import math
import random
from argparse import ArgumentParser
from qwerty import *
from qpe import qpe

def order_finding(err_tol, x, modN, acc=None):
    m = math.ceil(math.log2(modN))
    prec = 2*m + 1 + math.ceil(
        math.log2(2+1/(2*err_tol)))

    @qpu
    def one():
        return '0'**(m-1) * '1'

    @classical[[J]]
    @reversible
    def mult(y: bit[m]) -> bit[m]:
        return x**2**J * y % modN

    op = mult.inplace

    def run_qpe():
        angle_frac, = qpe(prec, one, op, shots=1, acc=acc).keys()
        return angle_frac

    frac1 = run_qpe()
    frac2 = run_qpe()

    def get_denom(frac):
        cf = cfrac(frac)
        for conv in reversed(cf.convergents()):
            if conv.denominator < modN:
                return conv.denominator

    return math.lcm(get_denom(frac1),
                    get_denom(frac2))

def shor(epsilon, num, acc=None):
    if num % 2 == 0:
        return 2

    while True:
        x = random.randint(2, num-1)
        if (y := math.gcd(x, num)) > 1:
            print('Got lucky! Skipping order subroutine')
            return y

        r = order_finding(epsilon, x, num, acc=acc)

        if r % 2 == 0 and pow(x, r//2, num) != -1:
            if (gcd := math.gcd(x**(r//2)-1, num)) > 1 \
                    and gcd != num:
                return gcd
            if (gcd := math.gcd(x**(r//2)+1, num)) > 1 \
                    and gcd != num:
                return gcd

        print("Shor's failed, trying again...")

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('number',
                        nargs='?',
                        default=15,
                        type=int,
                        help='The number to factor. Default: %(default)s')
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.2,
                        help='Error tolerance factor. Default: %(default)s')
    parser.add_argument('--acc', '-a',
                        default=None,
                        help='Name of an accelerator. The default is local '
                             'simulation.')
    args = parser.parse_args()

    num, epsilon = args.number, args.epsilon
    factor = shor(epsilon, num, acc=args.acc)
    print(f'Found factor of {num}: {factor}')
