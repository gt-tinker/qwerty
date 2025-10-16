"""
Qwerty demo of superdense coding, a technique for transmitting two classical
bits using one qubit, as described in Section 2.3 of Nielsen and Chuang.
"""

from argparse import ArgumentParser
from qwerty import *

def superdense_coding(payload: bit[2], shots: int):
    bit0, bit1 = payload

    @qpu
    def kernel():
        alice, bob = '00' + '11'

        sent_to_bob = (
            alice | ({'0'>>'1', '1'>>'0'}
                     if bit0 else id)
                  | ('1' >> -'1'
                     if bit1 else id))

        return (sent_to_bob * bob
                | {'00' + '11', '00' + -'11',
                   '10' + '01', '01' + -'10'}
                  .measure)

    return kernel(shots=shots)

def test(shots):
    for i in range(1 << 2):
        payload = bit[2](i)
        yield superdense_coding(payload, shots)
