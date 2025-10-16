"""
Showcases how the revolve basis generator works in the general (forward) case with phases. Here, the phase is within the revolve.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    return '0'**3 | ij**3 >> std**2 // {'0', '1'@25}.revolve | std.measure**3

def test(shots):
    return kernel(shots=shots)

