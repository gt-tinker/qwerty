"""
Similar to ``tilt.py`` except using symbols such as ``'i'`` and ``'j'``.
"""

from qwerty import *

@qpu
def kernel_p() -> bit:
    return 'p' | pm.measure

@qpu
def kernel_m() -> bit:
    return 'm' | pm.measure

@qpu
def kernel_i() -> bit:
    return 'i' | ij.measure

@qpu
def kernel_j() -> bit:
    return 'j' | ij.measure

def test(shots):
    return (kernel_p(shots=shots),
            kernel_m(shots=shots),
            kernel_i(shots=shots),
            kernel_j(shots=shots))
