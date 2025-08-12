"""
A trivial 'random bit' program that uses no metaQwerty features such as 'p'
"""

from qwerty import *

@qpu(prelude=None)
def kernel() -> bit:
    return __SYM_STD0__()+__SYM_STD1__() | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()})

def test(shots):
    return kernel(shots=shots)
