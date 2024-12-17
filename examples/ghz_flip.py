#!/usr/bin/env python3

"""
An alternate formulation of ghz.py. Again prepares an N-qubit
Greenberger–Horne–Zeilinger (GHZ) state (|00...0⟩ + |11...1⟩)/√2 for the
provided N and prints the measurement statistics.

This formulation demonstrates the `.flip' keyword and the predication
operator `&'. It may be run from the command line as well with the number of
qubits as an argument.
"""

from argparse import ArgumentParser
from qwerty import *

def ghz(n_qubits, acc=None):
    @qpu[[N]]
    def kernel() -> bit[N]:
        return '+' + '0'[N-1] | '1' & std.flip[N-1] \
                              | std[N].measure

    return kernel[[n_qubits]](histogram=True, shots=2048, acc=acc)

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

    print_histogram(ghz(args.num_qubits, acc=args.acc))
