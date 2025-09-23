from fractions import Fraction
from qwerty import *

def qpe(prec, get_init_state, op):
    @qpu[[M]]
    def kernel():
        return ('p'**prec * get_init_state()
                | (op[[prec-1-j]]
                   in '?'**j * '1' * '?'**(prec-1-j)
                      * '_'**M
                   for j in range(prec))
                | fourier[prec].measure
                  * discard**M)

    bits = kernel()
    return Fraction(int(bits),
                    2**len(bits))
