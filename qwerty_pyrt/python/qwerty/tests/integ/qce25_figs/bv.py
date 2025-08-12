"""
Fig. 14 of the Qwerty QCE '25 paper.
"""

from qwerty import *

def bv(secret_str):
  @classical[[N]]
  def f(x: bit[N]) -> bit:
    return (secret_str & x).xor_reduce()

  @qpu[[N]]
  def kernel() -> bit[N]:
    return ('p'**N | f.sign
                   | pm**N >> std**N
                   | measure**N)

  return kernel()

secret_str = bit[4](0b1101)
#print(bv(secret_str))

def test():
  return str(bv(secret_str))
