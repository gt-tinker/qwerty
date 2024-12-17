from qwerty import *
import qiskit.qasm3

def get_circuit(f):
  @qpu[[N]](f)
  def kernel(f: cfunc[N,1]) \
            -> bit[N]:
    return 'p'[N] | f.sign \
                  | pm[N] >> std[N] \
                  | std[N].measure

  qasm = kernel.qasm()
  return qiskit.qasm3.loads(qasm), qasm

def get_black_box(secret_string):
  @classical[[N]](secret_string)
  def f(secret_string: bit[N],
        x: bit[N]) -> bit:
    return (secret_string & x) \
           .xor_reduce()

  return f
