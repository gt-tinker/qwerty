#!/usr/bin/env python3

"""
A Qwerty implementation of the Stern–Gerlach experiment as described by Karam [1].
(Karam in turn cites Sakurai, McIntyre et al., and Townsend).
This is primarily intended to be run from the command line.

[1] R. Karam, “Why are complex numbers needed in quantum mechanics? Some answers for the
    introductory level,” American Journal of Physics, vol. 88, no. 1, pp.
    39–45, Jan. 2020, doi: 10.1119/10.0000258.
"""

from argparse import ArgumentParser
from qwerty import *

def stern_gerlach(n_shots, acc=None):
    @qpu
    def experiment1() -> bit:
        return '0' | pm.project | std.measure

    print('Experiment 1:')
    print_histogram(experiment1(shots=n_shots, histogram=True, acc=acc))

    @qpu
    def experiment2() -> bit:
        return '0' | ij.project | std.measure

    print('Experiment 2:')
    print_histogram(experiment2(shots=n_shots, histogram=True, acc=acc))

    @qpu
    def experiment3() -> bit:
        return 'p' | ij.project | ij.measure

    print('Experiment 3:')
    print_histogram(experiment3(shots=n_shots, histogram=True, acc=acc))

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()
    stern_gerlach(n_shots=4096, acc=args.acc)
