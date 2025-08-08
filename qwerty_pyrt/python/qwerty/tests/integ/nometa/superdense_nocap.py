"""
A version of the superdense coding example from the QCE '25 paper without any
captures or metaQwerty features.
"""

from qwerty import *

@qpu
def kernel00() -> bit[2]:
    payload = bit[2](0b00)
    bit0, bit1 = payload

    alice, bob = __SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__()

    id = __SYM_PAD__() >> __SYM_PAD__()
    sent_to_bob = (
        alice | ({__SYM_STD0__()>>__SYM_STD1__(), __SYM_STD1__()>>__SYM_STD0__()}
                 if bit0 else id)
              | (__SYM_STD1__() >> -__SYM_STD1__()
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {__SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD0__() + -__SYM_STD1__()*__SYM_STD1__(),
                __SYM_STD1__()*__SYM_STD0__() + __SYM_STD0__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD1__() + -__SYM_STD1__()*__SYM_STD0__()}))

@qpu
def kernel01() -> bit[2]:
    payload = bit[2](0b01)
    bit0, bit1 = payload

    alice, bob = __SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__()

    id = __SYM_PAD__() >> __SYM_PAD__()
    sent_to_bob = (
        alice | ({__SYM_STD0__()>>__SYM_STD1__(), __SYM_STD1__()>>__SYM_STD0__()}
                 if bit0 else id)
              | (__SYM_STD1__() >> -__SYM_STD1__()
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {__SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD0__() + -__SYM_STD1__()*__SYM_STD1__(),
                __SYM_STD1__()*__SYM_STD0__() + __SYM_STD0__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD1__() + -__SYM_STD1__()*__SYM_STD0__()}))

@qpu
def kernel10() -> bit[2]:
    payload = bit[2](0b10)
    bit0, bit1 = payload

    alice, bob = __SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__()

    id = __SYM_PAD__() >> __SYM_PAD__()
    sent_to_bob = (
        alice | ({__SYM_STD0__()>>__SYM_STD1__(), __SYM_STD1__()>>__SYM_STD0__()}
                 if bit0 else id)
              | (__SYM_STD1__() >> -__SYM_STD1__()
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {__SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD0__() + -__SYM_STD1__()*__SYM_STD1__(),
                __SYM_STD1__()*__SYM_STD0__() + __SYM_STD0__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD1__() + -__SYM_STD1__()*__SYM_STD0__()}))
@qpu
def kernel11() -> bit[2]:
    payload = bit[2](0b11)
    bit0, bit1 = payload

    alice, bob = __SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__()

    id = __SYM_PAD__() >> __SYM_PAD__()
    sent_to_bob = (
        alice | ({__SYM_STD0__()>>__SYM_STD1__(), __SYM_STD1__()>>__SYM_STD0__()}
                 if bit0 else id)
              | (__SYM_STD1__() >> -__SYM_STD1__()
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {__SYM_STD0__()*__SYM_STD0__() + __SYM_STD1__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD0__() + -__SYM_STD1__()*__SYM_STD1__(),
                __SYM_STD1__()*__SYM_STD0__() + __SYM_STD0__()*__SYM_STD1__(), __SYM_STD0__()*__SYM_STD1__() + -__SYM_STD1__()*__SYM_STD0__()}))

def test(shots):
    return kernel00(shots=shots), kernel01(shots=shots), kernel10(shots=shots), kernel11(shots=shots)
