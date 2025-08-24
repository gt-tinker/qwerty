#!/usr/bin/env python3

"""
Implementation of period finding as defined in Section 5.4.1 in Nielsen and
Chuang. Uses the implementation of phase estimation from qpe.py.

When run directly, this module also acts as a tester for the Qwerty
implementation of period finding. The function whose period is found takes an
M-bit input and clears all bits except the lower K bits — that is, it takes the
modulus mod 2^K. Thus, the correct period is 2^K. The number of input bits,
output bits, and K (the number of lower bits to keep) are passed on the command
line. The result from the algorithm is then checked for correctness.
"""

import math
from argparse import ArgumentParser
from qwerty import *

def period_finding(black_box, acc=None):
    @qpu[[M,N]](black_box)
    def kernel(black_box: cfunc[M,N]) -> bit[M]:
        return 'p'[M] + '0'[N] | black_box.xor \
                               | fourier[M].measure + discard[N]

    result1, result2 = kernel(shots=2, acc=acc)
    l_over_r1 = result1.as_bin_frac()
    l_over_r2 = result2.as_bin_frac()
    r = math.lcm(l_over_r1.denominator, l_over_r2.denominator)
    return r

def get_black_box(n_bits_in, n_bits_out, n_mask_bits):
    @classical[[M,N,K]]
    def f(x: bit[M]) -> bit[N]:
        return bit[N-K](0b0), x[M-K:]

    return f[[n_bits_in, n_bits_out, n_mask_bits]]

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('num_bits_in',
                        type=int,
                        help='Number of input bits to the black box. '
                             'Example: 4')
    parser.add_argument('num_bits_out',
                        type=int,
                        help='Number of output bits to the black box. '
                             'Example: 3')
    parser.add_argument('num_masked_bits',
                        type=int,
                        help='The number of bits K to keep in the output. '
                             'That is, set the resulting period to 2^K. '
                             'Example: 2')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')

    args = parser.parse_args()
    black_box = get_black_box(args.num_bits_in,
                              args.num_bits_out,
                              args.num_masked_bits)

    if period_finding(black_box, acc=args.acc) == 2**args.num_masked_bits:
        print('Success!')
    else:
        print('Period finding failed. Please try again.')
