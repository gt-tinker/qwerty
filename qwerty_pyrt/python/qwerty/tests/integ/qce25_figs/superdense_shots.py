"""
Fig. 5 of the Qwerty QCE '25 paper, except taking a shots arg.
"""

from qwerty import *

def superdense_coding(payload: bit[2], shots: int):
  bit0, bit1 = payload

  @qpu
  def kernel() -> bit[2]:
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

def test(payload, shots):
  return superdense_coding(payload, shots)
