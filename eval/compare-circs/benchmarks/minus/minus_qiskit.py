import math
import qiskit.qasm2
from qiskit import QuantumCircuit

def parity(x):
    out = 0
    while x:
        out ^= x & 1
        x >>= 1
    return out

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)
    circ.initialize([(-1.0 if parity(i) else 1.0) * math.sqrt(1/2**n_qubits) + 0.j for i in range(2**n_qubits)], range(n_qubits))
    circ.measure(range(n_qubits), range(n_qubits))
    # reps=6 should be enough in my testing, but let's burn some coal in Macon for fun
    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
