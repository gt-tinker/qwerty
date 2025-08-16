"""
Fig. 14 of the Qwerty QCE '25 paper, except taking a shots arg.
"""

from qwerty import *

def bv(secret_str, shots):
  @classical[[N]]
  def f(x: bit[N]) -> bit:
    return (secret_str & x).xor_reduce()

  @qpu[[N]]
  def kernel() -> bit[N]:
    return ('p'**N | f.sign
                   | pm**N >> std**N
                   | measure**N)

  return kernel(shots=shots)

secret_str = bit[4](0b1101)
#print(bv(secret_str))

def test(shots):
  return bv(secret_str, shots)
