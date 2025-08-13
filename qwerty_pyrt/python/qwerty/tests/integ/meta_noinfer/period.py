"""
Fig. 16 of the Qwerty QCE '25 paper, but reworked not to require type
inference.
"""

import math
from fractions import Fraction
from qwerty import *

def period_finding(f):
    @qpu
    def kernel() -> bit[3]:
        return ('p'**3 * '0'**3
                | f.xor
                | id**3 * discard**3
                | fourier[3].measure)
    
    def shift_binary_point(bits):
        return Fraction(int(bits),
                        2**len(bits))
    
    frac1 = shift_binary_point(kernel())
    frac2 = shift_binary_point(kernel())
    return math.lcm(frac1.denominator,
                    frac2.denominator)

@classical
def mod4(x: bit[3]) -> bit[3]:
    return x % 4

def test():
    return period_finding(mod4)
