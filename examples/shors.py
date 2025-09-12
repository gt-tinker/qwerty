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

from order_finding import order_finding

def shors(epsilon, num, acc=None):
    if num % 2 == 0:
        return 2

    x = random.randint(2, num-1)
    if (y := math.gcd(x, num)) > 1:
        print('Got lucky! Skipping quantum subroutine...')
        return y

    r = order_finding(epsilon, x, num, acc=acc)

    if r % 2 == 0 and pow(x, r//2, num) != -1:
        if (gcd := math.gcd(x**(r//2)-1, num)) > 1 \
                and gcd != num:
            return gcd
        if (gcd := math.gcd(x**(r//2)+1, num)) > 1 \
                and gcd != num:
            return gcd

    raise Exception("Shor's failed")

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('number',
                        type=int,
                        help='The number to factor. Example: 15')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    err = 0.2
    print('Nontrivial factor of', args.number, 'is',
          shors(err, args.number, acc=args.acc))
