#!/usr/bin/env python3

"""
Prepare an N-qubit Greenberger–Horne–Zeilinger (GHZ) state
(|00...0⟩ + |11...1⟩)/√2 for the provided N and print the measurement
statistics. May be run from the command line with the number of qubits as an
argument.
"""

from argparse import ArgumentParser
from qwerty import *

def ghz(num_qubits, shots=None, histogram=False, acc=None):
    @qpu[[N]]
    def kernel() -> bit[N]:
        return ('0'[N] or '1'[N]) | measure[N]

    return kernel[[num_qubits]](shots=shots, histogram=histogram, acc=acc)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_qubits',
                        type=int,
                        help='The size of the GHZ state to prepare (i.e., the '
                             'number of Qubits N)')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    print_histogram(ghz(args.num_qubits, shots=1024, histogram=True, acc=args.acc))
