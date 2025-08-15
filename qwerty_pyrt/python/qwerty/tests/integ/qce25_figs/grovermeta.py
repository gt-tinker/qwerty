"""
Fig. 9 of the Qwerty QCE '25 paper.
"""

from qwerty import *

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
  from .grover import oracle
  return grover2(oracle, 3)
