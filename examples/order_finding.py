#!/usr/bin/env python3

"""
A Qwerty implementation of quantum order finding as defined in Section 5.3.1 of
Nielsen and Chuang.

When run directly, this module also acts as a tester for the Qwerty
implementation of order finding. The x and N must be specified on the command line.
"""

import math
from argparse import ArgumentParser
from qwerty import *

from qpe import qpe

def order_finding(epsilon, x, modN, acc=None):
    L = math.ceil(math.log2(modN))
    t = 2*L + 1 + math.ceil(math.log2(2+1/(2*epsilon)))

    @qpu[[M]]
    def one() -> qubit[M]:
        return '0'[M-1] + '1'

    @classical[[X,N,M,J]]
    def xymodN(y: bit[M]) -> bit[M]:
        return X**2**J * y % N

    x_inv = pow(x, -1, modN)
    fwd = xymodN[[x,modN,L,...]]
    rev = xymodN[[x_inv,modN,L,...]]
    mult = fwd.inplace(rev)
    frac1, frac2 = qpe(t, one, mult, 2, acc=acc)

    def denom(frac):
        cf = cfrac.from_fraction(frac)
        for c in reversed(cf.convergents()):
            if c.denominator < modN:
                return c.denominator

    return math.lcm(denom(frac1), denom(frac2))

def naive_classical(x, modN):
    for r in range(1, modN):
        if x**r % modN == 1:
            return r

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('x',
                        type=int,
                        help='The integer x whose multiplicative order to '
                             'find. Example: 23')
    parser.add_argument('modN',
                        type=int,
                        help='The (integer) modulus N for which to find the '
                             'smallest positive integer r such that '
                             'x^r ≡ 1 (mod N). Example: 4')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    err = 0.2
    x = args.x
    modN = args.modN
    if math.gcd(x, modN) != 1:
        raise ValueError(f'{x} and {modN} are not relatively prime')

    print('Classical:', naive_classical(x, modN))
    print('Quantum:  ', order_finding(err, x, modN, acc=args.acc))
