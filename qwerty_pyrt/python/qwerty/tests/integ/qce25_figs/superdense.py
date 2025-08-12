"""
Fig. 5 of the Qwerty QCE '25 paper.
"""

def superdense_coding(payload: bit[2]):
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

  recovered_payload = kernel()
  return recovered_payload

def test():
  return superdense_coding(payload)
