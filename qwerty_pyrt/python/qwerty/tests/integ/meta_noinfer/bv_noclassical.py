"""
A version of Bernstein–Vazirani with metaQwerty features but no classical
function embeddings. This version only uses some qubit symbols, just as a test.
"""

from qwerty import *

@qpu
def kernel() -> bit[3]:
    # secret string is 110
    f_sign = {'0'*'1'*'0', '0'*'1'*'1', '1'*'0'*'0', '1'*'0'*'1'} >> {-'0'*'1'*'0', -'0'*'1'*'1', -'1'*'0'*'0', -'1'*'0'*'1'}
    return (('0'+'1')*('0'+'1')*('0'+'1')
            | f_sign
            | {'0'+'1','0'-'1'}*{'0'+'1','0'-'1'}*{'0'+'1','0'-'1'} >> {'0','1'}*{'0','1'}*{'0','1'}
            | {'0','1'}.measure*{'0','1'}.measure*{'0','1'}.measure)

def test(shots):
    return kernel(shots=shots)
