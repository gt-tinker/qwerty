"""
Showcases how the revolve basis generator works in the general (reverse) case.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    return '0'**3 | std**2 // pm.revolve >> ij**3 | std.measure**3

def test(shots):
    return kernel(shots=shots)

