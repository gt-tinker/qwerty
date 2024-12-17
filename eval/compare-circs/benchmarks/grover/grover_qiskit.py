# Reference code from qiskit-textbook
# https://github.com/qiskit-community/qiskit-textbook/blob/main/content/ch-algorithms/grover.ipynb

import qiskit.qasm2
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal

def diffuser(qc, n_qubits):
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(n_qubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(n_qubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(n_qubits-1)
    qc.mcx(list(range(n_qubits-1)), n_qubits-1)  # multi-controlled-toffoli
    qc.h(n_qubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(n_qubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(n_qubits):
        qc.h(qubit)

def get_circuit(n_qubits, n_iter):
    grover_circuit = QuantumCircuit(n_qubits, n_qubits)
    grover_circuit.h(range(n_qubits))

    for _ in range(n_iter):
        # Oracle for 1111...1
        grover_circuit.h(n_qubits-1)
        grover_circuit.mcx(list(range(n_qubits-1)), n_qubits-1)
        grover_circuit.h(n_qubits-1)
        # Diffuser
        diffuser(grover_circuit, n_qubits)

    grover_circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return grover_circuit, qiskit.qasm2.dumps(grover_circuit)
