"""
Prepare a Bell state (a two-qubit entangled state) and print measurement
statistics.
"""

from argparse import ArgumentParser
from qwerty import *

@qpu
def kernel():
    return '00' + '11' | measure**2

def test(shots):
    return kernel(shots=shots)
