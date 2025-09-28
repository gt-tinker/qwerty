"""A test for the ``x.concat(y)`` construct in ``@classical`` functions`"""

from qwerty import *

@classical
def concat(x: bit[3], y: bit[2]) -> bit[5]:
    return x.concat(y)

@qpu
def kernel():
    return '010'*'01' * '0'**5 | concat.xor | measure**10

def test(shots):
    return (concat(bit[3](0b010), bit[2](0b01)), kernel(shots=shots))
