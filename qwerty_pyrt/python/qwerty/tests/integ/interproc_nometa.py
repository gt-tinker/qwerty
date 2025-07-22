"""
A trivial 'random bit' program that calls one kernel from another (but uses no
metaQwerty features such as 'p').
"""

from qwerty import *

@qpu
def get_p() -> qubit:
    return '0'+'1'

@qpu
def kernel() -> bit:
    return get_p() | __MEASURE__({'0','1'})

def test(shots):
    return kernel(shots=shots)
