"""
A version of Bernstein–Vazirani with metaQwerty features and no captures in
``@classical` functions. (That is, the secret string is hard-coded as a bit
literal.) This version also does not use broadcast tensors, just as a test.
"""

from qwerty import *

@classical
def oracle(x: bit[3]) -> bit:
    return (x & bit[3](0b110)).xor_reduce()

@qpu
def kernel() -> bit[3]:
    return ('ppp'
            | oracle.sign
            | {'p','m'}*{'p','m'}*{'p','m'} >> {'0','1'}*{'0','1'}*{'0','1'}
            | ({'0','1'}*{'0','1'}*{'0','1'}).measure)

def test(shots):
    return kernel(shots=shots)
