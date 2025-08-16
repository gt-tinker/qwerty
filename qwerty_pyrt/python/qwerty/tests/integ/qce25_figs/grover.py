"""
Fig. 2 of the Qwerty QCE '25 paper. (Fig. 1 is an expression from this
program.)
"""

from qwerty import *

@classical
def oracle(x: bit[4]) -> bit:
  return x[0] & ~x[1] & x[2] & ~x[3]

@qpu
def grover_iter(q: qubit[4]):
  return (q | oracle.sign
            | 'pppp' >> -'pppp')

@qpu
def grover():
  return (
    'pppp' | grover_iter
           | grover_iter
           | grover_iter
           | measure**4)

#print(grover())

def test_runs():
  return str(grover())

def test_correct(shots):
  return grover(shots=shots)
