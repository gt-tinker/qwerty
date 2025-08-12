"""
A tiny example of a ``@classical`` function.
"""

from qwerty import *

@classical(prelude=None)
@reversible
def flip(x: bit[3]) -> bit[3]:
    return ~x

@qpu(prelude=None)
def kernel() -> bit[3]:
    return (__SYM_STD0__()*__SYM_STD0__()*__SYM_STD0__()
            | __EMBED_INPLACE__(flip)
            | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}))

def test(shots):
    return kernel(shots=shots)
