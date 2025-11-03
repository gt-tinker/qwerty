import qiskit.qasm

from qwerty import *

def get_circuit(n_qubits):
    @qpu
    def kernel():
        return '0'**n_qubits | ij**n_qubits >> std**(n_qubits - 1) // {'0', '1'@25}.revolve | measure**n_qubits

    qasm = kernel.qasm()
    return qiskit.qasm3.loads(qasm), qasm
