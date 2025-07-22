"""
A simple example of taking tensor product of functions without any metaQwerty.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return '0'*('0'+'1') | ({'0','1'} >> {'1','0'}) * __DISCARD__() | [] * __MEASURE__({'0','1'}) * []

def test(shots):
    return kernel(shots=shots)
