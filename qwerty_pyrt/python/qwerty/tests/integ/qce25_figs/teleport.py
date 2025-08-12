"""
Fig. 13 of the Qwerty QCE '25 paper.
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
def kernel_0() -> bit:
  return '0' | teleport | std.measure

@qpu
def kernel_1() -> bit:
  return '1' | teleport | std.measure

@qpu
def kernel_p() -> bit:
  return 'p' | teleport | pm.measure

@qpu
def kernel_m() -> bit:
  return 'm' | teleport | pm.measure

@qpu
def kernel_i() -> bit:
  return 'i' | teleport | ij.measure

@qpu
def kernel_j() -> bit:
  return 'j' | teleport | ij.measure

def test(shots):
  return (kernel_0(shots=shots),
          kernel_1(shots=shots),
          kernel_p(shots=shots),
          kernel_m(shots=shots),
          kernel_i(shots=shots),
          kernel_j(shots=shots))
