"""
Test Deutsch's algorithm.
"""

from qwerty import *

def deutsch(f, shots=None):
    @qpu
    def kernel():
        return 'p' | f.sign | pm.measure

    return kernel(shots=shots)

@classical
def balanced(x: bit) -> bit:
    return ~x

@classical
def constant(x: bit) -> bit:
    return bit[1](0b1)

def naive_classical(f):
    return f(bit[1](0b0)) ^ f(bit[1](0b1))

def test(shots):
    return [(naive_classical(balanced), deutsch(balanced, shots=shots)),
            (naive_classical(constant), deutsch(constant, shots=shots))]
