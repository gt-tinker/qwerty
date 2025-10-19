import qiskit.qasm3

from qwerty import *

def get_circuit(n_qubits):
    @qpu[[N,J]]
    def kernel() -> bit[N]:
        return ['0' or '1' @ (360 * ((J % 2**(k+1)) / 2**(k+1)))
                for k in range(N)] \
               | measure[N]

    # Prepare the jth Fourier basis state as defined by Nielsen
    # and Chuang (Chapter 5)
    j = -1 & ~(-1 << n_qubits)
    qasm = kernel[[n_qubits, j]].qasm()
    return qiskit.qasm3.loads(qasm), qasm
