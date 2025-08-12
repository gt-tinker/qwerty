"""
A tiny example of a ``@classical`` function that uses metaQwerty.
"""

from qwerty import *

@classical
@reversible
def flip(x: bit[3]) -> bit[3]:
    return ~x

@qpu
def kernel() -> bit[3]:
    return ('0'**3
            | flip.inplace
            | measure**3)

def test(shots):
    return kernel(shots=shots)
