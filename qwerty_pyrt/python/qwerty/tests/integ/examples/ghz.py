"""
Prepare an N-qubit Greenberger–Horne–Zeilinger (GHZ) state, a Bell
state generalized to N qubits, and print measurement statistics. May be run
from the command line with the number of qubits as an argument.
"""

from qwerty import *

def test(num_qubits, shots):
    @qpu
    def kernel():
        return '0'**num_qubits + '1'**num_qubits | measure**num_qubits

    return kernel(shots=shots)
