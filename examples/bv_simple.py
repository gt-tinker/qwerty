#!/usr/bin/env python3

"""
A simpler version of bv.py without the classical algorithm. Runs the
Bernstein—Vazirani algorithm on a provided bitstring and prints the bitstring
found by the algorithm.
"""

from qwerty import *

def bv(secret_string, acc=None):
    @classical[[N]](secret_string)
    def f(secret_string: bit[N], x: bit[N]) -> bit:
        return (secret_string & x).xor_reduce()

    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] | f.sign \
                      | pm[N] >> std[N] \
                      | measure[N]

    return kernel(acc=acc)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('secret_bits',
                        help='The secret bitstring used to define the black '
                             'box for f(x) (i.e., the oracle). Example: 1101')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    secret_string = bit.from_str(args.secret_bits)
    print(bv(secret_string, args.acc))
