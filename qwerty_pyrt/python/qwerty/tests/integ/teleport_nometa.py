"""
A version of quantum teleportation with no metaQwerty features.
"""

from qwerty import *

@qpu
def teleport(payload: qubit) -> qubit:
  alice, bob = '00' + '11'

  id = '?' >> '?'
  flip = {'0'>>'1', '1'>>'0'}

  bit_flip, sign_flip = (
    alice * payload | (flip if '_1' else id)
                    | __MEASURE__({'0','1'} * {'0'+'1','0'-'1'}))

  teleported_payload = (
    bob | (flip if bit_flip else id)
        | ('1' >> -'1'
           if sign_flip else id))

  return teleported_payload

@qpu
def kernel_0() -> bit:
    return '0' | teleport | __MEASURE__({'0','1'})

@qpu
def kernel_1() -> bit:
    return '1' | teleport | __MEASURE__({'0','1'})

@qpu
def kernel_p() -> bit:
    return '0'+'1' | teleport | __MEASURE__({'0'+'1','0'-'1'})

@qpu
def kernel_m() -> bit:
    return '0'-'1' | teleport | __MEASURE__({'0'+'1','0'-'1'})

def test(shots):
    return kernel_0(shots=shots), kernel_1(shots=shots), kernel_p(shots=shots), kernel_m(shots=shots)
