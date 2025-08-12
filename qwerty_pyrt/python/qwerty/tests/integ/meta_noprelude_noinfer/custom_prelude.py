"""
Tests that a custom prelude with custom vector symbols (including unicode
symbols) works.
"""

from qwerty import *

@qpu_prelude
def custom_prelude():
    '↑'.sym = __SYM_STD0__()
    '↓'.sym = __SYM_STD1__()
    '+'.sym = '↑' + '↓'
    '-'.sym = '↑' - '↓'
    'L'.sym = '↑' + '↓'@90
    'R'.sym = '↑' + '↓'@270

    up_down = {'↑', '↓'}
    plus_minus = {'+', '-'}
    left_right = {'L', 'R'}

    b.M = __MEASURE__(b)

@qpu(prelude=custom_prelude)
def kernel_0() -> bit:
    return '↑' | up_down.M

@qpu(prelude=custom_prelude)
def kernel_1() -> bit:
    return '↓' | up_down.M

@qpu(prelude=custom_prelude)
def kernel_p() -> bit:
    return '+' | plus_minus.M

@qpu(prelude=custom_prelude)
def kernel_m() -> bit:
    return '-' | plus_minus.M

@qpu(prelude=custom_prelude)
def kernel_i() -> bit:
    return 'L' | left_right.M

@qpu(prelude=custom_prelude)
def kernel_j() -> bit:
    return 'R' | left_right.M

def test(shots):
    return (kernel_0(shots=shots),
            kernel_1(shots=shots),
            kernel_p(shots=shots),
            kernel_m(shots=shots),
            kernel_i(shots=shots),
            kernel_j(shots=shots))
