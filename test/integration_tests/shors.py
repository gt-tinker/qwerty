from qwerty import *
import random
import math

from .order_finding import order_finding

def shors(number):
    if not (number & 0x1):
        return 2

    #x = random.randint(2, number-1)
    # Hardcode an x to make this test more deterministic
    x = 7
    if (gcd := math.gcd(x, number)) > 1:
        return gcd

    order = order_finding(0.2, x, number)
    if not (order & 0x1) and x**(order//2) % number != -1:
        if (gcd := math.gcd(x**(order//2)-1, number)) > 1 \
                and gcd != number:
            return gcd
        if (gcd := math.gcd(x**(order//2)+1, number)) > 1 \
                and gcd != number:
            return gcd

    raise ValueError('need to retry shors')

def test(number):
    return shors(number)

