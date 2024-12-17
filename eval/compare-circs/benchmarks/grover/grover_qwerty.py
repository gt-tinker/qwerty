import math
from qwerty import *
import qiskit.qasm3

def get_n_iter(n_qubits, n_answers):
  n = 2**n_qubits
  m = n_answers
  theta = 2*math.acos(
            math.sqrt((n-m)/n))
  rnd = lambda x: math.ceil(x-0.5)
  return rnd(math.acos(
             math.sqrt(m/n))/theta)

# For our tested input
@classical[[N]]
def all_ones(x: bit[N]) -> bit:
  return x.and_reduce()

def grovers(oracle, n_iter):
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
  qasm = kernel[[n_iter]].qasm()
  return qiskit.qasm3.loads(qasm), qasm 

def get_circuit(n_qubits, n_iter):
    return grovers(all_ones[[n_qubits]], n_iter)
