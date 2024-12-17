import qiskit.qasm2
from qiskit import QuantumCircuit

def constant(qubits):
    oracle = QuantumCircuit(qubits + 1)
    oracle.x(qubits)
    return oracle

def balanced(qubits):
    oracle = QuantumCircuit(qubits + 1)
    for i in range(qubits):
        oracle.cx(i, qubits)
    return oracle

def get_circuit(oracle):
    n = oracle.num_qubits - 1
    dj_circuit = QuantumCircuit(n+1, n)

    dj_circuit.x(n)
    for qubit in range(n+1):
        dj_circuit.h(qubit)
    dj_circuit.append(oracle, range(n + 1))
    for qubit in range(n):
        dj_circuit.h(qubit)

    for i in range(n):
        dj_circuit.measure(i, i)

    return dj_circuit, qiskit.qasm2.dumps(dj_circuit)

