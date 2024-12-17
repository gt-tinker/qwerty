#!/usr/bin/env python3

"""
Implementation of fixed-point amplitude amplification as proposed by Yoder et
al. [1]. Expects a function `a' as input that produces the search space given
qubits in the |0⟩ state as input, called `A' in [1]; a classical function
`oracle' that returns 1 on a matching N-bit standard basis state, a specific
form of `U' from [1]; and a worst-case answer probability, called `λ` in [1].

When run directly, this module also acts as a tester for the Qwerty
implementation. Given a number of bits N as input, it runs a fixed-point search
across the all 2^N-bit standard basis states for a bitstring where the first
N-1 bits are _not_ all 1s. This means 111...10 and 111...11 should be left with
near-zero probability.

[1] T. J. Yoder, G. H. Low, and I. L. Chuang, “Fixed-Point Quantum Search with
    an Optimal Number of Queries,” Phys. Rev. Lett., vol. 113, no. 21, p.
    210501, Nov. 2014, doi: 10.1103/PhysRevLett.113.210501.
"""

from argparse import ArgumentParser
from qwerty import *
from fix_pt_phases import get_phases

def fix_pt_amp(a, oracle, orig_prob,
               new_prob=0.98, n_shots=2048, histogram=False, acc=None):
    phis = get_phases(orig_prob, new_prob)

    @qpu[[N,K,D]](phis, a, oracle)
    def amp_iter(phis: angle[2*D], a: rev_qfunc[N], oracle: cfunc[N,1],
                 q: qubit[N+1]) -> qubit[N+1]:
        return q | oracle.xor \
                 | id[N] + std.rotate(phis[[2*K]]) \
                 | oracle.xor \
                 | ~a + id \
                 | '0'[N] & std.flip \
                 | id[N] + std.rotate(phis[[2*K+1]]) \
                 | '0'[N] & std.flip \
                 | a + id

    @qpu[[N,D]](phis, a, amp_iter)
    def kernel(phis: angle[2*D], a: qfunc[N],
               amp_iter: qfunc[N+1][[...]]) -> bit[N]:
        return '0'[N+1] | a + id \
                        | (amp_iter[[k]] for k in range(D)) \
                        | std[N].measure + discard

    return kernel(shots=n_shots, histogram=histogram, acc=acc)

if __name__ == '__main__':
    @qpu[[N]]
    @reversible
    def a(q: qubit[N]) -> qubit[N]:
        return q | 'p'[N].prep

    @classical[[N]]
    def oracle_(x: bit[N]) -> bit:
        return ~x[:N-1].and_reduce()

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('n_qubits',
                        type=int,
                        help='The number of qubits N on which to run '
                             'fixed-point search.')
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()
    n_qubits = args.n_qubits
    oracle = oracle_[[n_qubits]]
    # Conservative probability guess
    orig_prob = 1/2**n_qubits
    res = fix_pt_amp(a, oracle, orig_prob, 0.98, histogram=True, acc=args.acc)
    print_histogram(res)
