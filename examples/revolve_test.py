#!/usr/bin/env python3

from argparse import ArgumentParser
from qwerty import *

@qpu
def kernel():
    return '0'**3 | ij**3 >> std**2 // {'0', '1'@25}.revolve | measure**3

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--shots', '-s',
                        type=int,
                        default=1024,
                        help='Number of shots. Default: %(default)s')
    parser.add_argument('--acc', '-a',
                        default=None,
                        help='Name of an accelerator. The default is local '
                             'simulation.')
    args = parser.parse_args()

    print('Results:')
    histogram(kernel(shots=args.shots, acc=args.acc))
