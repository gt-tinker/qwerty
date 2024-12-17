import sys
from qwerty import *

def bv(secret_string):
    @classical[[N]](secret_string)
    def f(secret_string: bit[N], x: bit[N]) -> bit:
        return (secret_string & x).xor_reduce()

    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] + 'm' | f.xor        \
                            | id[N] + discard[1] \
                            | pm[N] >> std[N]    \
                            | std[N].measure

    return kernel()
