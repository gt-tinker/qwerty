#!/usr/bin/env python3

import math
import random
from qwerty import *
from .qpe import qpe

def order_finding(err_tol, x, modN):
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
        angle_frac, = qpe(prec, one, op, shots=1).keys()
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

def shor(epsilon, num, num_attempts=32):
    if num % 2 == 0:
        return 2

    for _ in range(num_attempts):
        x = random.randint(2, num-1)
        if (y := math.gcd(x, num)) > 1:
            print('got lucky! retrying to test order finding (this is expected)...')
            continue

        r = order_finding(epsilon, x, num)

        if r % 2 == 0 and pow(x, r//2, num) != -1:
            if (gcd := math.gcd(x**(r//2)-1, num)) > 1 \
                    and gcd != num:
                return gcd
            if (gcd := math.gcd(x**(r//2)+1, num)) > 1 \
                    and gcd != num:
                return gcd

        print('retrying (this is expected)...')
    else:
        raise Exception(f'exceeded {num_attempts} tries')

def test(number, num_attempts):
    epsilon = 0.2
    return shor(epsilon, number, num_attempts)
