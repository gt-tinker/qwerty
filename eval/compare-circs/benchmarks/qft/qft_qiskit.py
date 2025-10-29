import math
import cmath
import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)

    # Equation (5.2) from Nielsen & Chuang – prepare the j-th Fourier basis state
    # (left commented out because it reinitializes the circuit state)
    # circ.initialize([
    #     cmath.exp(2j * math.pi * j * k / (1 << n_qubits)) / math.sqrt(1 << n_qubits)
    #     for k in range(1 << n_qubits)
    # ])

    # TODO: Clean this up
    for k in range(n_qubits):
        j = n_qubits - k  # qubit index counting from MSB to LSB

        # Hadamard on qubit j-1
        circ.h(j - 1)

        # Conditional rotations with lower significance
        for i in reversed(range(j - 1)):
            circ.cp(2 * math.pi / 2 ** (j - i), i, j - 1)

    # Swap qubits to reverse order
    for i in range(n_qubits // 2):
        circ.swap(i, n_qubits - i - 1)

    # Measure all qubits into classical bits
    circ.measure(range(n_qubits), range(n_qubits))

    # Optional: decompose to primitive gates
    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
