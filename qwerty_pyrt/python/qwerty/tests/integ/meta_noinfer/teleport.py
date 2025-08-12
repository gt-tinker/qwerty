"""
A version of quantum teleportation with metaQwerty features.
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
        | ('1' >> -'1' if sign_flip else id))

  return teleported_payload

@qpu
def kernel_0() -> bit:
    return '0' | teleport | std.measure

@qpu
def kernel_1() -> bit:
    return '1' | teleport | std.measure

@qpu
def kernel_p() -> bit:
    return '0'+'1' | teleport | pm.measure

@qpu
def kernel_m() -> bit:
    return '0'-'1' | teleport | pm.measure

def test(shots):
    return (kernel_0(shots=shots),
            kernel_1(shots=shots),
            kernel_p(shots=shots),
            kernel_m(shots=shots))
