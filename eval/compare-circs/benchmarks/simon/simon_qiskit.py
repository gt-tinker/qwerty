import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(n_qubits):
    simon_circuit = QuantumCircuit(2*n_qubits, n_qubits)

    # initialize inputs in state |+>
    simon_circuit.h(range(n_qubits))

    # Apply our secret sauce oracle
    k = n_qubits // 2
    # Copy first k inputs to outputs
    simon_circuit.cx(range(k), range(n_qubits, n_qubits + k))
    # Copy last k-1 inputs to last k-1 outputs
    simon_circuit.cx(range(k+1, n_qubits), range(n_qubits + k+1, 2*n_qubits))
    # XOR input k with the last k-1 outputs in-place
    simon_circuit.cx([k for _ in range(k-1)], range(n_qubits + k+1, 2*n_qubits))

    simon_circuit.h(range(n_qubits))
    simon_circuit.measure(range(n_qubits), range(n_qubits))

    return simon_circuit, qiskit.qasm2.dumps(simon_circuit)
