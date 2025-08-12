"""
A trivial 'random bit' program that calls one kernel from another and uses
metaQwerty features such as ``'p'``.
"""

from qwerty import *

@qpu
def get_p() -> qubit:
    return 'p'

@qpu
def kernel() -> bit:
    return measure(get_p())

def test(shots):
    return kernel(shots=shots)
