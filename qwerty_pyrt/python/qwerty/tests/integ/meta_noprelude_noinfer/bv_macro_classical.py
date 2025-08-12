"""
A version of Bernstein–Vazirani with metaQwerty macros but without classical
function embeddings.
"""

from qwerty import *

secret_string = bit[3](0b110)

@classical(prelude=None)
def oracle(x: bit[3]) -> bit:
    return (x & secret_string).xor_reduce()

@qpu(prelude=None)
def kernel() -> bit[3]:
    '0'.sym = __SYM_STD0__()
    '1'.sym = __SYM_STD1__()
    'p'.sym = '0'+'1'
    'm'.sym = '0'-'1'
    std = {'0','1'}
    pm = {'p','m'}
    b.measure = __MEASURE__(b)
    measure = std.measure
    f.expr.sign = __EMBED_SIGN__(f)

    return ('p'**3
            | oracle.sign
            | pm**3 >> std**3
            | measure**3)

def test(shots):
    return kernel(shots=shots)
