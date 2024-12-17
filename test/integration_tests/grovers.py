import sys
import math
from qwerty import *

def grover(oracle, n_iter, n_shots):
  @qpu[[N]](oracle)
  def grover_iter(oracle: cfunc[N,1],
                  q: qubit[N]) \
                 -> qubit[N]:
    return q | oracle.sign \
             | -('p'[N] >> -'p'[N])

  @qpu[[N,I]](grover_iter)
  def kernel(grover_iter: qfunc[N]) \
            -> bit[N]:
    return \
      'p'[N] | (grover_iter
                for _ in range(I)) \
             | std[N].measure

  kern_inst = kernel[[n_iter]]
  results = kern_inst(shots=n_shots)
  return {r for r in set(results)
            if oracle(r)}

def get_n_iter(n_qubits, n_answers):
  n = 2**n_qubits
  m = n_answers
  theta = 2*math.acos(
            math.sqrt((n-m)/n))
  rnd = lambda x: math.ceil(x-0.5)
  return rnd(math.acos(
             math.sqrt(m/n))/theta)

# Flag bitstrings of all ones as the answer
@classical[[N]]
def all_ones(x: bit[N]) -> bit:
    return x.and_reduce()

def test(n_qubits):
    n_ans = 1
    oracle = all_ones[[n_qubits]]
    n_iter = get_n_iter(n_qubits, n_ans)
    answers = grover(oracle, n_iter,
                    n_shots=32)
    return answers
