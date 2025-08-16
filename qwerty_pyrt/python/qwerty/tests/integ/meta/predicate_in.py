"""
Test the `in` syntactic sugar for predication.
"""

from qwerty import *

@qpu
def kernel():
    return '111' | (flip in '1_1') | measure**3

def test(shots):
    return kernel(shots=shots)
