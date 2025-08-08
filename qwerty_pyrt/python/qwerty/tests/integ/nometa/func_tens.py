"""
A simple example of taking tensor product of functions without any metaQwerty.
"""

from qwerty import *

@qpu
def kernel() -> bit:
    return __SYM_STD0__()*(__SYM_STD0__()+__SYM_STD1__()) | ({__SYM_STD0__(),__SYM_STD1__()} >> {__SYM_STD1__(),__SYM_STD0__()}) * __DISCARD__() | [] * __MEASURE__({__SYM_STD0__(),__SYM_STD1__()}) * []

def test(shots):
    return kernel(shots=shots)
