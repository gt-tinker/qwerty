"""
Fig. 9 of the Qwerty QCE '25 paper.
"""

from qwerty import *

@classical
def oracle(x: bit[4]) -> bit:
  return x[0] & ~x[1] & x[2] & ~x[3]

def grover2(oracle, num_iter):
  @qpu[[N]]
  def grover_iter(q):
    return (q | oracle.sign
              | 'p'**N >> -'p'**N)

  @qpu[[N]]
  def kernel():
    return (
      'p'**N | (grover_iter
                for i in range(num_iter))
             | measure**N)

  return kernel()

def test():
  return grover2(oracle, 3)
