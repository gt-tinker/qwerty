"""
A trivial 'random bit' program that uses metaQwerty features.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return measure('p')

def test(shots):
    return kernel(shots=shots)
