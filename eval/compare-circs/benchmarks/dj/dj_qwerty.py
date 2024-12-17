from qwerty import *
import qiskit.qasm3

def get_circuit(f):
  @qpu[[N]](f)
  def kernel(f: cfunc[N,1]) \
            -> bit[N]:
    return 'p'[N] | f.sign \
                  | pm[N].measure
  qasm = kernel.qasm()
  return qiskit.qasm3.loads(qasm), qasm

@classical[[N]]
def constant(x: bit[N]) -> bit:
  # f(x) = 1
  return bit[1](0b1)

@classical[[N]]
def balanced(x: bit[N]) -> bit:
  # f(x) = 1 for half the inputs
  # and f(x) = 0 for the other half
  return x.xor_reduce()
