"""
A trivial 'random bit' program that uses no metaQwerty features such as 'p'
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return '0'+'1' | __MEASURE__({'0','1'})

def test(shots):
    return kernel(shots=shots)
