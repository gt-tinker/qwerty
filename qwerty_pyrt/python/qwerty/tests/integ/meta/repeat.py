"""A test for the ``x.repeat(N)`` construct in ``@classical`` functions`"""

from qwerty import *

@classical
def repeat4(x: bit):
    return x.repeat(4)

@qpu
def kernel():
    return '1' * '0'**4 | repeat4.xor | measure**5

def test(shots):
    return (repeat4(bit[1](0b1)), kernel(shots=shots))
