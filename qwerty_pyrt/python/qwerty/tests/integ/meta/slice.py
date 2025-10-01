"""
A test for slicing (``x[lo:hi]``) in ``@classical`` functions`

Classically, the call ``slice(bit[5](0b01_110))`` should return
``bit[5](0b01_110)``. Thus, an XOR embedding of ``slice`` taking
``'01'*'110`*'00000'` as input should produce ``'01'*'110'*'01110'`` because
``bit[5](0b00_000) ^ bit[5](0b01_110) == bit[5](0b01_110)``.
"""

from qwerty import *

@classical
def slice(x: bit[5]) -> bit[5]:
    return x[:2].concat(x[2:])

@qpu
def kernel():
    return '01'*'110' * '0'**5 | slice.xor | measure**10

def test(shots):
    return (slice(bit[5](0b01_110)), kernel(shots=shots))
