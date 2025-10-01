"""
A version of Bernstein–Vazirani that uses type inference and macros.
"""

from qwerty import *

secret_string = bit[4](0b1101)

@classical[[N]]
def oracle(x: bit[N]) -> bit:
    return (x & secret_string).xor_reduce()

@qpu[[N]]
def kernel():
    return ('p'**N | oracle.sign
                   | pm**N >> std**N
                   | measure**N)

def test(shots):
    return kernel(shots=shots)
