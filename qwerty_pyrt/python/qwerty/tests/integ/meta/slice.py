"""A test for slicing ``x[lo:hi]`` in ``@classical`` functions`"""

from qwerty import *

@classical
def slice(x: bit[5]) -> bit[5]:
    return x[:2].concat(x[2:])

@qpu
def kernel():
    return '011'*'10' * '0'**5 | slice.xor | measure**10

def test(shots):
    return (slice(bit[5](0b011_10)), kernel(shots=shots))
