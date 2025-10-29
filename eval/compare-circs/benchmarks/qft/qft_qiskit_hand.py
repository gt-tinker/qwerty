import qiskit.qasm2
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)

    circ.append(QFT(num_qubits=n_qubits), range(n_qubits))
    circ.measure(range(n_qubits), range(n_qubits))

    # reps=6 should be enough in my testing, but let's burn some coal in Macon for fun
    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
