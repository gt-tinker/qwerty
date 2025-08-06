"""
A version of Bernstein–Vazirani with neither metaQwerty features nor classical
function embeddings.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    # secret string is 110
    f_sign = {'010', '011', '100', '101'} >> {-'010', -'011', -'100', -'101'}
    return (('0'+'1')*('0'+'1')*('0'+'1')
            | f_sign
            | {'0'+'1','0'-'1'}*{'0'+'1','0'-'1'}*{'0'+'1','0'-'1'} >> {'0','1'}*{'0','1'}*{'0','1'}
            | __MEASURE__({'0','1'}*{'0','1'}*{'0','1'}))

def test(shots):
    return kernel(shots=shots)
