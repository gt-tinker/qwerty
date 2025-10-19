import qiskit.qasm3

from qwerty import *
from .vectors import vecs

def get_circuit(n_qubits):
    vec = vecs[n_qubits]

    @qpu[[N(n_qubits),M]](vec)
    def kernel(vec: ampl[M]) -> bit[N]:
        return vec.q | measure[N]

    qasm = kernel.qasm()
    return qiskit.qasm3.loads(qasm), qasm
