"""
A simple example of taking tensor product of functions with metaQwerty.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return '0p' | flip * discard | [] * measure * []

def test(shots):
    return kernel(shots=shots)
