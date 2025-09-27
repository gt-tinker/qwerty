#!/usr/bin/env python3

"""
Prepare an N-qubit Greenberger–Horne–Zeilinger (GHZ) state, a Bell
state generalized to N qubits, and print measurement statistics. May be run
from the command line with the number of qubits as an argument.
"""

from argparse import ArgumentParser
from qwerty import *

def ghz(num_qubits, shots):
    @qpu
    def kernel():
        return '0'**num_qubits + '1'**num_qubits | measure**num_qubits

    return kernel(shots=shots)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_qubits',
                        type=int,
                        help='The size of the GHZ state to prepare (i.e., the '
                             'number of qubits)')
    parser.add_argument('--shots', '-s',
                        type=int,
                        default=1024,
                        help='Number of shots'
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    histogram(ghz(args.num_qubits, args.shots))
