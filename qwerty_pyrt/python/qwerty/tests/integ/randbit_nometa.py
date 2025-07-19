"""
A trivial 'random bit' program that uses no metaQwerty features such as 'p'
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return '0'+'1' | {'0','1'}.measure

def test(shots):
    return kernel(shots=shots)
