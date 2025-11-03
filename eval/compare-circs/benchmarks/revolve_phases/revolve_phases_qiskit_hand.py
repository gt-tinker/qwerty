import math
import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)

    for i in range(n_qubits):
        circ.sdg(i)
        circ.h(i)

    circ.h(0)

    for i in range(1, n_qubits):
        angle = math.pi / (2 ** i)
        circ.cp(angle, i, 0)  # control=i, target=0

    final_angle = math.pi * 25.0 / 180.0
    circ.rz(final_angle, 0)

    circ.measure(range(n_qubits), range(n_qubits))

    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
