"""
A trivial Bell state program that uses no metaQwerty features. Intended to test
the lowering of the ``Predicated`` AST node.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    id = __SYM_PAD__() >> __SYM_PAD__()
    flip = {__SYM_STD0__()>>__SYM_STD1__(), __SYM_STD1__()>>__SYM_STD0__()}
    return (__SYM_STD0__()+__SYM_STD1__())+__SYM_STD0__() | (flip if __SYM_STD1__()*__STD__TARGET__() else id) | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()})

def test(shots):
    return kernel(shots=shots)
