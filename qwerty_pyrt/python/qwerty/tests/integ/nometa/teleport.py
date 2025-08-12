"""
A version of quantum teleportation with no metaQwerty features.
"""

from qwerty import *

@qpu(prelude=None)
def teleport(payload: qubit) -> qubit:
  alice, bob = __SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__()

  id = __SYM_PAD__() >> __SYM_PAD__()
  flip = {__SYM_STD0__()>>__SYM_STD1__(), __SYM_STD1__()>>__SYM_STD0__()}

  bit_flip, sign_flip = (
    alice * payload | (flip if __SYM_TARGET__()*__SYM_STD1__() else id)
                    | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()} * {__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()}))

  teleported_payload = (
    bob | (flip if bit_flip else id)
        | (__SYM_STD1__() >> -__SYM_STD1__()
           if sign_flip else id))

  return teleported_payload

@qpu(prelude=None)
def kernel_0() -> bit:
    return __SYM_STD0__() | teleport | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()})

@qpu(prelude=None)
def kernel_1() -> bit:
    return __SYM_STD1__() | teleport | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()})

@qpu(prelude=None)
def kernel_p() -> bit:
    return __SYM_STD0__()+__SYM_STD1__() | teleport | __MEASURE__({__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()})

@qpu(prelude=None)
def kernel_m() -> bit:
    return __SYM_STD0__()-__SYM_STD1__() | teleport | __MEASURE__({__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()})

def test(shots):
    return kernel_0(shots=shots), kernel_1(shots=shots), kernel_p(shots=shots), kernel_m(shots=shots)
