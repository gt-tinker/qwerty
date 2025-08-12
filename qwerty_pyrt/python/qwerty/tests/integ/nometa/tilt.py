"""
Some Qwerty test programs involving the tilt operator ``@``.
"""

from qwerty import *

@qpu(prelude=None)
def kernel_p() -> bit:
    return __SYM_STD0__() + __SYM_STD1__() | __MEASURE__({__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()+__SYM_STD1__()@180})

@qpu(prelude=None)
def kernel_m() -> bit:
    return __SYM_STD0__() + __SYM_STD1__()@180 | __MEASURE__({__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()+__SYM_STD1__()@180})

@qpu(prelude=None)
def kernel_i() -> bit:
    return __SYM_STD0__() + __SYM_STD1__()@90 | __MEASURE__({__SYM_STD0__()+__SYM_STD1__()@90, __SYM_STD0__()+__SYM_STD1__()@270})

@qpu(prelude=None)
def kernel_j() -> bit:
    return __SYM_STD0__() + __SYM_STD1__()@270 | __MEASURE__({__SYM_STD0__()+__SYM_STD1__()@90, __SYM_STD0__()+__SYM_STD1__()@270})

def test(shots):
    return (kernel_p(shots=shots),
            kernel_m(shots=shots),
            kernel_i(shots=shots),
            kernel_j(shots=shots))
