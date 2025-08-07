"""
A tiny example of a ``@classical`` function.
"""

from qwerty import *

@classical
@reversible
def flip(x: bit[3]) -> bit[3]:
    return ~x

@qpu
def kernel() -> bit[3]:
    return ('000'
            | __EMBED_INPLACE__(flip)
            | __MEASURE__({'0','1'}*{'0','1'}*{'0','1'}))

def test(shots):
    return kernel(shots=shots)
