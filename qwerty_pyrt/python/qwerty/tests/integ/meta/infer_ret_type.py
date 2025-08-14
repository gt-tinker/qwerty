"""
Test inferring the return type of a ``@qpu`` kernel.
"""

from qwerty import *

@qpu
def kernel():
    return '111' | (std**3).measure

def test(shots):
    return kernel(shots=shots)
