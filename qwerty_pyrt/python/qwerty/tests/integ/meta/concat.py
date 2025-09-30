"""
A test for the ``x.concat(y)`` construct in ``@classical`` functions`.

Classically, the call ``concat32(bit[3](0b010), bit[2](0b01))`` should return
``bit[5](0b010_01)``. Thus, an XOR embedding of ``concat32`` taking
``'010'*'01`*'00000'` as input should produce ``'010'*'01'*'01001'`` because
``bit[5](0b000_00) ^ bit[5](0b010_01) == bit[5](0b010_01)``.
"""

from qwerty import *

@classical
def concat32(x: bit[3], y: bit[2]) -> bit[5]:
    return x.concat(y)

@qpu
def kernel():
    return '010'*'01' * '0'**5 | concat32.xor | measure**10

def test(shots):
    return (concat32(bit[3](0b010), bit[2](0b01)), kernel(shots=shots))
