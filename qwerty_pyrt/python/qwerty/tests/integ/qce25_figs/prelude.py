"""
Figs. 6 and 7 of the Qwerty QCE '25 paper as a ``@qpu_prelude`` block and used
in a simple Qwerty program.
"""

from qwerty import *

@qpu_prelude
def qce25_prelude():
    # Symbols in qubit literals
    '0'.sym = __SYM_STD0__()
    '1'.sym = __SYM_STD1__()
    'p'.sym = '0' + '1'
    'i'.sym = '0' + '1'@90
    'm'.sym = '0' + '1'@180
    'j'.sym = '0' + '1'@270

    # Common bases
    std = {'0', '1'}
    pm = {'p', 'm'}
    ij = {'i', 'j'}
    bell = {'00' + '11', '00' + -'11',
            '10' + '01', '01' + -'10'}

    # Basis macros
    {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    b.measure = __MEASURE__(b)
    {bv1, bv2}.revolve = __REVOLVE__(bv1, bv2)

    # More complicated bases
    fourier[1] = pm
    fourier[N] = fourier[N-1] // std.revolve

    # Built-in functions
    id = {'0','1'} >> {'0','1'}
    discard = __DISCARD__()
    flip = std.flip
    measure = std.measure

@qpu(prelude=qce25_prelude)
def kernel() -> bit:
    return 'm' | pm.measure

def test(shots):
    return kernel(shots=shots)
