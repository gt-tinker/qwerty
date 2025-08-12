"""
A version of Bernstein–Vazirani with neither metaQwerty features beyond
the most trivial (``**``/``'0'``) nor classical function embeddings.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    '0'.sym = __SYM_STD0__()
    '1'.sym = __SYM_STD1__()
    'p'.sym = '0'+'1'
    'm'.sym = '0'-'1'

    # secret string is 110
    f_sign = {'010', '011', '100', '101'} >> {-'010', -'011', -'100', -'101'}
    return ('p'**3
            | f_sign
            | {'p','m'}**3 >> {'0','1'}**3
            | __MEASURE__({'0','1'}**3))

def test(shots):
    return kernel(shots=shots)
