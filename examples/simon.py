#!/usr/bin/env python3

"""
A Qwerty implementation of Simon's algorithm, the first quantum algorithm to
promise exponential speedup.

When run directly, this module also acts as a tester for the Qwerty
implementation of Simon's. The number of qubits passed on the command line and
the secret string is printed out. The black box is a simple construction to
guarantee a particular 2-to-1 mapping as required by Simon's with a nonzero
secret string.
"""

from argparse import ArgumentParser
from qwerty import *

from simon_postprocess import simon_post, Retry

def simon(f, acc=None):
    @qpu[[N]]
    def kernel():
        return ('p'**N * '0'**N | f.xor
                                | (std**N >> pm**N) * id**N
                                | measure**N * discard**N)

    while True:
        rows = []
        while True:
            row = kernel(acc=acc)
            if int(row) != 0:
                rows.append(row)
                if len(rows) >= row.n_bits-1:
                    break
        try:
            return simon_post(rows)
        except Retry:
            print('retrying...')
            continue

def get_black_box(num_bits):
    """
    Returns a simple function that meets the requirements of Simon's algorithm.
    Specifically, the function must be 2-to-1 such that ``f(x) == f(y)`` if and
    only if `x == y` or ``x == y ^ s`` where ``s`` is a nonzero secret
    bitstring.

    The function defined here is ``f(xy) = 0y`` where ``x` is 1 bit and ``y``
    is ``num_bits-1`` bits. This is 2-to-1 because ``f(1y) == f(0y)``, and
    the secret string ``s`` is ``100...0`` since ``1y == 0y ^ s`` as needed.
    """
    @classical
    def black_box(x: bit[num_bits]) -> bit[num_bits]:
        return bit[1](0b0).concat(x[1:])

    return black_box

def naive_classical(f, num_bits):
    out_to_x = {}
    for i in range(2**num_bits):
        x = bit[num_bits](i)
        out = f(x)
        if out in out_to_x:
            return x ^ out_to_x[out]
        out_to_x[out] = x

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_bits',
                        type=int,
                        help='The length of the secret string')
    parser.add_argument('--acc', '-a',
                        default=None,
                        help='Name of an accelerator. The default is local '
                             'simulation.')
    args = parser.parse_args()

    num_bits = args.num_bits
    f = get_black_box(num_bits)

    print('Classical:', naive_classical(f, num_bits))
    print('Quantum:  ', simon(f, acc=args.acc))
