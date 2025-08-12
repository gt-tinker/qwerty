"""
A trivial 'random bit' program that calls one kernel from another (but uses no
metaQwerty features such as 'p').
"""

from qwerty import *

@qpu(prelude=None)
def get_p() -> qubit:
    return __SYM_STD0__()+__SYM_STD1__()

@qpu(prelude=None)
def kernel() -> bit:
    return get_p() | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()})

def test(shots):
    return kernel(shots=shots)
