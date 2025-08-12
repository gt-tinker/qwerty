"""
A version of the superdense coding example from the QCE '25 paper with captures
and metaQwerty features, including ``id`` and ``flip``.
"""

from qwerty import *

def superdense_coding(payload, shots=None):
    @qpu
    def kernel() -> bit[2]:
        bit0, bit1 = payload
        alice, bob = '00' + '11'

        sent_to_bob = (
            alice | (flip if bit0 else id)
                  | ('1' >> -'1'
                     if bit1 else id))

        return (sent_to_bob * bob
                | {'00' + '11', '00' + -'11',
                   '10' + '01', '01' + -'10'}.measure)

    return kernel(shots=shots)

def test(shots):
    return (superdense_coding(bit[2](0b00), shots=shots),
            superdense_coding(bit[2](0b01), shots=shots),
            superdense_coding(bit[2](0b10), shots=shots),
            superdense_coding(bit[2](0b11), shots=shots))
