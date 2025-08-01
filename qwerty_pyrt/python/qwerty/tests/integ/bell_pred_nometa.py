"""
A trivial Bell state program that uses no metaQwerty features. Intended to test
the lowering of the ``Predicated`` AST node.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    id = '?' >> '?'
    flip = {'0'>>'1', '1'>>'0'}
    return ('0'+'1')+'0' | (flip if '1_' else id) | __MEASURE__({'0','1'})

def test(shots):
    return kernel(shots=shots)
