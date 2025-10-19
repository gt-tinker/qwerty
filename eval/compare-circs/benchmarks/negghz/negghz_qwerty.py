import qiskit.qasm3

from qwerty import *

def get_circuit(n_qubits):
    @qpu[[N]]
    def kernel() -> bit[N]:
        return ('0'[N] or -'1'[N]) | measure[N]

    qasm = kernel[[n_qubits]].qasm()
    return qiskit.qasm3.loads(qasm), qasm
