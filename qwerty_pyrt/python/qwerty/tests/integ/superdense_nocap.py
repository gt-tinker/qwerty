"""
A version of the superdense coding example from the QCE '25 paper without any
captures or metaQwerty features.
"""

from qwerty import *

@qpu
def kernel00() -> bit[2]:
    alice, bob = '00' + '11'

    id = '?' >> '?'
    sent_to_bob = (
        alice | ({'0'>>'1', '1'>>'0'}
                 if bit[1](0b0) else id)
              | ('1' >> -'1'
                 if bit[1](0b0) else id))

    return (sent_to_bob * bob
            | __MEASURE__(
               {'00' + '11', '00' + -'11',
                '10' + '01', '01' + -'10'}))
#@qpu
#def kernel01() -> bit[2]:
#    alice, bob = '00' + '11'
#
#    sent_to_bob = (
#        alice | ({'0'>>'1', '1'>>'0'}
#                 if bit[1](0b0) else id)
#              | ('1' >> -'1'
#                 if bit1[1](0b1) else id))
#
#    return (sent_to_bob * bob
#            | {'00' + '11', '00' + -'11',
#               '10' + '01', '01' + -'10'}
#              .measure)
#

def test(shots):
    return kernel00(shots=shots)
