"""
A version of Bernstein–Vazirani with metaQwerty features.
"""

from qwerty import *

secret_string = bit[3](0b110)

@classical
def oracle(x: bit[3]) -> bit:
    return (x & secret_string).xor_reduce()

@qpu
def kernel() -> bit[3]:
    return ('p'**3
            | oracle.sign
            | pm**3 >> std**3
            | measure**3)

def test(shots):
    return kernel(shots=shots)
