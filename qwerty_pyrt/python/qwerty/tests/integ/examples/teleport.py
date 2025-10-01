"""
Qwerty demo of quantum teleportation as described in Section 1.3.7 of Nielsen
and Chuang. Intended to be run from the command line.
"""

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

def test(shots):
    return (teleport_1(shots=shots),
            teleport_i(shots=shots))
