"""
A version of Bernstein–Vazirani with no metaQwerty features and no captures in
``@classical` functions. (That is, the secret string is hard-coded as a bit
literal.)
"""

from qwerty import *

@classical
def oracle(x: bit[3]) -> bit:
    return (x & bit[3](0b110)).xor_reduce()

@qpu
def kernel() -> bit[3]:
    return (('0'+'1')*('0'+'1')*('0'+'1')
            | __EMBED_SIGN__(oracle)
            | {'0'+'1','0'-'1'}*{'0'+'1','0'-'1'}*{'0'+'1','0'-'1'} >> {'0','1'}*{'0','1'}*{'0','1'}
            | __MEASURE__({'0','1'}*{'0','1'}*{'0','1'}))

def test(shots):
    return kernel(shots=shots)
