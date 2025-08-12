"""
Prepares the Fourier basis state from Fig. 12 of the QCE '25 paper and measures
in the Fourier basis using metaQwerty.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    return ('mi' * ('0' + '1'@225)
            | fourier[3].measure)

def test(shots):
    return kernel(shots=shots)
