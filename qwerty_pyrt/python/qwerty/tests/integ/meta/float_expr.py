"""
Test a very basic ``FloatExpr``.
"""

from qwerty import *

@qpu
def kernel_const_expr():
    return 'p' | '1' >> '1'@(360.0/2) | pm.measure

angle_deg = 360.0

@qpu
def kernel_capture():
    return 'p' | '1' >> '1'@(angle_deg/2) | pm.measure

silly_angle_deg = 2880.0

@qpu
def kernel_pow_expr():
    return 'p' | '1' >> '1'@(silly_angle_deg/2**4) | pm.measure

def test(shots):
    return (kernel_const_expr(shots=shots),
            kernel_capture(shots=shots),
            kernel_pow_expr(shots=shots))
