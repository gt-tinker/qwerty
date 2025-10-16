"""
Test a simple case of QASM generation.
"""

from argparse import ArgumentParser
from qwerty import *

def ghz(num_qubits):
    @qpu
    def kernel():
        return '0'**num_qubits + '1'**num_qubits | measure**num_qubits

    return kernel.qasm()

def test(num_qubits):
    return ghz(num_qubits)
