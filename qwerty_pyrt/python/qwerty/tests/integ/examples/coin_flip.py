"""
A 'hello world' program that performs a coin flip 1024 times by repeatedly
measuring a fresh qubit in a uniform superposition of 0 and 1.
"""

from argparse import ArgumentParser
from qwerty import *

@qpu
def kernel():
    return '0'+'1' | measure

def test(shots):
    return kernel(shots=shots)
