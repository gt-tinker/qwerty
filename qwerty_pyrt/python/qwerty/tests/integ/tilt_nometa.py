"""
Some Qwerty test programs involving the tilt operator ``@``.
"""

from qwerty import *

@qpu
def kernel_p() -> bit:
    return '0' + '1' | __MEASURE__({'0'+'1','0'+'1'@180})

@qpu
def kernel_m() -> bit:
    return '0' + '1'@180 | __MEASURE__({'0'+'1','0'+'1'@180})

@qpu
def kernel_i() -> bit:
    return '0' + '1'@90 | __MEASURE__({'0'+'1'@90, '0'+'1'@270})

@qpu
def kernel_j() -> bit:
    return '0' + '1'@270 | __MEASURE__({'0'+'1'@90, '0'+'1'@270})

def test(shots):
    return (kernel_p(shots=shots),
            kernel_m(shots=shots),
            kernel_i(shots=shots),
            kernel_j(shots=shots))
