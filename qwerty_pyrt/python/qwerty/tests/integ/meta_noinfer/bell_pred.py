"""
A trivial Bell state program that uses metaQwerty. Intended to test the
lowering of the ``Predicated`` AST node.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return 'p0' | (flip if '1_' else id) | std.measure

def test(shots):
    return kernel(shots=shots)
