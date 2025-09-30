"""
A test for the ``x.repeat(N)`` construct in ``@classical`` functions`.

Classically, the call ``repeat4(bit[1](0b1))`` should return
``bit[4](0b1111)``. Thus, an XOR embedding of ``repeat4`` taking
``'1'*'0000`` as input should produce ``'1'*'1111'`` because
``bit[4](0b0000) ^ bit[4](0b1111) == bit[4](0b1111)``.
"""

from qwerty import *

@classical
def repeat4(x: bit):
    return x.repeat(4)

@qpu
def kernel():
    return '1' * '0'**4 | repeat4.xor | measure**5

def test(shots):
    return (repeat4(bit[1](0b1)), kernel(shots=shots))
