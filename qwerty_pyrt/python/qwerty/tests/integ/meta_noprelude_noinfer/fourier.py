"""
Prepares the Fourier basis state from Fig. 12 of the QCE '25
paper and measures in the Fourier basis using metaQwerty
features.
"""

from qwerty import *

@qpu(prelude=None)
def kernel() -> bit[3]:
    '0'.sym = __SYM_STD0__()
    '1'.sym = __SYM_STD1__()
    'p'.sym = '0' + '1'
    'm'.sym = '0' + '1'@180
    'i'.sym = '0' + '1'@90
    b.measure = __MEASURE__(b)
    std = {'0','1'}
    pm = {'p','m'}
    {bv1, bv2}.revolve = __REVOLVE__(bv1, bv2)
    fourier[1] = pm
    fourier[N] = fourier[N-1] // std.revolve

    return ('mi' * ('0' + '1'@225)
            | fourier[3].measure)

def test(shots):
    return kernel(shots=shots)
