"""
An adaption of Fig. 2 of the Qwerty QCE '25 paper that does not need type
inference.
"""

from qwerty import *

@classical
def oracle(x: bit[4]) -> bit:
  return x[0] & ~x[1] & x[2] & ~x[3]

@qpu
def grover_iter(q: qubit[4]) -> qubit[4]:
  return (q | oracle.sign
            | 'pppp' >> -'pppp')

@qpu
def grover() -> bit[4]:
  return (
    'pppp' | grover_iter
           | grover_iter
           | grover_iter
           | measure**4)

def test(shots):
  return grover(shots=shots)
