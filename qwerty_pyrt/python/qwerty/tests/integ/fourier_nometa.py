"""
Prepares the Fourier basis state from Fig. 12 of the QCE '25 paper and measures
in the Fourier basis, all without metaQwerty features.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    return (('0' + '1'@180) * ('0' + '1'@90) * ('0' + '1'@225)
            | __MEASURE__(({'0'+'1', '0'-'1'} // __REVOLVE__('0','1')) // __REVOLVE__('0','1')))

def test(shots):
    return kernel(shots=shots)
