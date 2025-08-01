"""
A version of the superdense coding example from the QCE '25 paper without any
captures or metaQwerty features.
"""

from qwerty import *

@qpu
def kernel00() -> bit[2]:
    payload = bit[2](0b00)
    bit0, bit1 = payload

    alice, bob = '00' + '11'

    id = '?' >> '?'
    sent_to_bob = (
        alice | ({'0'>>'1', '1'>>'0'}
                 if bit0 else id)
              | ('1' >> -'1'
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {'00' + '11', '00' + -'11',
                '10' + '01', '01' + -'10'}))

@qpu
def kernel01() -> bit[2]:
    payload = bit[2](0b01)
    bit0, bit1 = payload

    alice, bob = '00' + '11'

    id = '?' >> '?'
    sent_to_bob = (
        alice | ({'0'>>'1', '1'>>'0'}
                 if bit0 else id)
              | ('1' >> -'1'
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {'00' + '11', '00' + -'11',
                '10' + '01', '01' + -'10'}))

@qpu
def kernel10() -> bit[2]:
    payload = bit[2](0b10)
    bit0, bit1 = payload

    alice, bob = '00' + '11'

    id = '?' >> '?'
    sent_to_bob = (
        alice | ({'0'>>'1', '1'>>'0'}
                 if bit0 else id)
              | ('1' >> -'1'
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {'00' + '11', '00' + -'11',
                '10' + '01', '01' + -'10'}))
@qpu
def kernel11() -> bit[2]:
    payload = bit[2](0b11)
    bit0, bit1 = payload

    alice, bob = '00' + '11'

    id = '?' >> '?'
    sent_to_bob = (
        alice | ({'0'>>'1', '1'>>'0'}
                 if bit0 else id)
              | ('1' >> -'1'
                 if bit1 else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {'00' + '11', '00' + -'11',
                '10' + '01', '01' + -'10'}))

def test(shots):
    return kernel00(shots=shots), kernel01(shots=shots), kernel10(shots=shots), kernel11(shots=shots)
