"""
Qwerty demo of quantum teleportation as described in Section 1.3.7 of Nielsen
and Chuang. Intended to be run from the command line.
"""

from argparse import ArgumentParser
from qwerty import *

@qpu
def teleport(payload: qubit) -> qubit:
    alice, bob = '00' + '11'

    bit_flip, sign_flip = (
        alice * payload | (flip if '_1' else id)
                        | (std * pm).measure)

    teleported_payload = (
        bob | (flip if bit_flip else id)
            | ('1' >> -'1'
               if sign_flip else id))

    return teleported_payload

@qpu
def teleport_1():
    alice = '1'
    bob = teleport(alice)
    return std.measure(bob)

@qpu
def teleport_i():
    alice = 'i'
    bob = teleport(alice)
    return ij.measure(bob)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--shots', '-s',
                        type=int,
                        default=1024,
                        help='Number of shots. Default: %(default)s')
    args = parser.parse_args()

    print("Teleporting a '1' state from Alice to Bob."
          "\nWhen Bob measures in the standard basis, he should always get a 1 bit:")
    histogram(teleport_1(shots=args.shots))
    print("\nTeleporting an 'i' state from Alice to Bob."
          "\nWhen Bob measures in the ij basis, he should always get a 0 bit"
          "\n(since 'i' has index 0 in {'i','j'}):")
    histogram(teleport_i(shots=args.shots))
