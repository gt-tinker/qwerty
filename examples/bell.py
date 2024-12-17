#!/usr/bin/env python3

"""
Prepare a Bell state (|00⟩ + |11⟩)/√2 and print the measurement statistics.
"""

from argparse import ArgumentParser
from qwerty import *

@qpu
def kernel() -> bit[2]:
    return ('00' or '11') | measure[2]

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    print_histogram(kernel(shots=1024, histogram=True, acc=args.acc))
