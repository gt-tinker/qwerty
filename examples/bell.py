#!/usr/bin/env python3

"""
Prepare a Bell state (a two-qubit entangled state) and print measurement
statistics.
"""

from argparse import ArgumentParser
from qwerty import *

@qpu
def kernel():
    return '00' + '11' | measure**2

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--shots', '-s',
                        type=int,
                        default=1024,
                        help='Number of shots. Default: %(default)s')
    args = parser.parse_args()

    histogram(kernel(shots=args.shots))
