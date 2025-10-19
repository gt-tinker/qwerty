import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)
    circ.h(0)
    if n_qubits > 1:
        circ.cx(range(n_qubits-1), range(1, n_qubits))
    circ.z(0)
    circ.measure(range(n_qubits), range(n_qubits))
    # reps=6 should be enough in my testing, but let's burn some coal in Macon for fun
    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
