#!/usr/bin/env python3

"""
A Qwerty implementation of Simon's algorithm, the first quantum algorithm to
promise exponential speedup.

When run directly, this module also acts as a tester for the Qwerty
implementation of Simon's. The number of qubits passed on the command line and
the secret string is printed out. The black box is a custom construction to
guarantee a particular 2-to-1 mapping as required by Simon's with a nonzero
secret string.
"""

from argparse import ArgumentParser
from qwerty import *

from simon_post import simon_post, Retry

def simon(f, acc=None):
    @qpu[[N]](f)
    def kernel(f: cfunc[N]) -> bit[N]:
        return 'p'[N] + '0'[N] | f.xor \
                               | (std[N] >> pm[N]) + id[N] \
                               | std[N].measure + discard[N]

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

@classical[[K]]
def black_box(x: bit[2*K]) -> bit[2*K]:
    return x[:K], bit[1](0b0), (x[K].repeat(K-1) ^ x[K+1:])

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_bits',
                        type=int,
                        help='The length of the secret string. Must be even. '
                             'Example: 6')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    n_bits = args.num_bits
    f = black_box[[n_bits//2]]

    def naive_classical(f, n_bits):
        out_to_x = {}
        for i in range(2**n_bits):
            x = bit[n_bits](i)
            out = f(x)
            if out in out_to_x:
                return x ^ out_to_x[out]
            out_to_x[out] = x

    print('Classical:', naive_classical(f, n_bits))
    print('Quantum:  ', simon(f, acc=args.acc))
