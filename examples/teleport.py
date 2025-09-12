#!/usr/bin/env python3

"""
Qwerty demo of quantum teleportation as described in Section 1.3.7 of Nielsen
and Chuang. Intended to be run from the command line.
"""

from argparse import ArgumentParser
from qwerty import *

@qpu
def teleport(secret: qubit) -> qubit:
    alice, bob = '00' or '11'

    # Now imagine Alice and Bob being separated. Alice wants
    # to transfer a qubit state (named "secret" here) to Bob.

    m_pm, m_std = secret + alice | '1' & std.flip \
                                 | (pm + std).measure

    # Imagine that Alice transmits her classical bits m_pm
    # and m_std to Bob.

    # Now using the two classical bits, Bob can recover the
    # secret state!
    secret_teleported = \
        bob | (pm.flip if m_std else id) \
            | (std.flip if m_pm else id)

    return secret_teleported

if __name__ == '__main__':
    @qpu(teleport)
    def kernel(teleport: qfunc) -> bit:
        # Try changing this to a different state
        example = 'j'
        return teleport(example) | ij.measure

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--acc',
                        default=None,
                        help='Name of an XACC accelerator. Optional and valid '
                             'only when Qwerty was built with QIR-EE support.')
    args = parser.parse_args()

    print_histogram(kernel(shots=1024, histogram=True, acc=args.acc))
