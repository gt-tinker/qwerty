import qiskit.qasm3

from qwerty import *

def get_circuit(n_qubits):
    @qpu
    def kernel():
        return '0'**n_qubits | std**n_qubits >> std**(n_qubits - 1) // std.revolve | measure**n_qubits

    qasm = kernel.qasm()
    return qiskit.qasm3.loads(qasm), qasm
