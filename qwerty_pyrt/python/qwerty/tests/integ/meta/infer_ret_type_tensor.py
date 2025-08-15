"""
Test inferring the return type of a ``@qpu`` kernel, except with a trickier
function tensor product involved.
"""

from qwerty import *

@qpu
def kernel():
    return '111' | measure**3

def test(shots):
    return kernel(shots=shots)
