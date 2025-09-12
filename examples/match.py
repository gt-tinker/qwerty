#!/usr/bin/env python3

"""
Implementation of the Niroula–Nam string matching algorithm [1] in Qwerty.
Takes the needle and haystack as input and returns matching position indices.
Uses fixed-point amplitude amplification (fix_pt_amp.py) for amplitude
amplification.

When run directly, this module also acts as a tester for the Qwerty
implementation of string matching. The string (haystack) and pattern (needle)
must be passed on the command line. Matching indices are printed.

[1] P. Niroula and Y. Nam, “A quantum algorithm for string matching,” npj
    Quantum Inf, vol. 7, no. 1, Art. no. 1, Feb. 2021, doi:
    10.1038/s41534-021-00369-3.
"""

import math
from argparse import ArgumentParser
from qwerty import *

from fix_pt_amp import fix_pt_amp

def match(string, pat, acc=None):
    n, m = len(string), len(pat)
    k = math.ceil(math.log2(n))

    @classical[[K(k),N(n),M(m)]]
    def shift_and_cmp(off: bit[K], string: bit[N], pat: bit[M]) -> bit[K+N+M]:
        return off, string, (string.rotl(off)[:M] ^ pat)

    @qpu[[K(k),N,M]](string, pat, shift_and_cmp)
    @reversible
    def a(string: bit[N], pat: bit[M], shift_and_cmp: cfunc[K+N+M],
                q: qubit[K+N+M]) -> qubit[K+N+M]:
        return q | 'p'[K].prep + string.prep + pat.prep \
                 | shift_and_cmp.inplace(shift_and_cmp)

    @classical[[K(k),N(n),M(m)]]
    def oracle(off: bit[K], string: bit[N], pat: bit[M]) -> bit:
        return (~pat).and_reduce()

    ret = fix_pt_amp(a, oracle, 1/n, acc=acc)
    return {int(result[:k]) for result in set(ret) if oracle(result)}

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('string',
                        help='The bitstring to search (i.e., the haystack). '
                             'Length must be a power of 2. Example: 1110')
    parser.add_argument('pattern',
                        help='The bitstring to find (i.e., the needle). '
                             'Length must be a power of 2. Example: 10')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()
    string = bit.from_str(args.string)
    pattern = bit.from_str(args.pattern)

    print('Matching indices:')
    for index in match(string, pattern, acc=args.acc):
        print(index)
