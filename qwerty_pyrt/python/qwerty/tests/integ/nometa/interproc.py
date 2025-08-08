"""
A trivial 'random bit' program that calls one kernel from another (but uses no
metaQwerty features such as 'p').
"""

from qwerty import *

@qpu
def get_p() -> qubit:
    return __SYM_STD0__()+__SYM_STD1__()

@qpu
def kernel() -> bit:
    return get_p() | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()})

def test(shots):
    return kernel(shots=shots)
