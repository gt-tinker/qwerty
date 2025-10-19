import math
import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)
    circ.initialize([1/math.sqrt(2) + 0.j] + [0.0j]*(2**n_qubits - 2) + [-1/math.sqrt(2) + 0.j])
    circ.measure(range(n_qubits), range(n_qubits))
    # reps=6 should be enough in my testing, but let's burn some coal in Macon for fun
    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
