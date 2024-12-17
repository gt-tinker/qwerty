# Reference code from qiskit-textbook
# https://github.com/qiskit-community/qiskit-textbook/blob/main/content/ch-algorithms/bernstein-vazirani.ipynb
import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(qubits, secret_string):
    bv_circuit = QuantumCircuit(qubits+1, qubits)

    # put auxiliary in state |->
    bv_circuit.h(qubits)
    bv_circuit.z(qubits)

    # Apply Hadamard gates before querying the oracle
    for i in range(qubits):
        bv_circuit.h(i)
        
    # Apply barrier -> No need
    # bv_circuit.barrier()

    # Apply the inner-product oracle
    secret_string = secret_string[::-1] # reverse secret_string to fit qiskit's qubit ordering
    for q in range(qubits):
        if secret_string[q] == '0':
            bv_circuit.id(q)
        else:
            bv_circuit.cx(q, qubits)
            
    # Apply barrier -> No need
    # bv_circuit.barrier()

    #Apply Hadamard gates after querying the oracle
    for i in range(qubits):
        bv_circuit.h(i)

    # Measurement
    for i in range(qubits):
        bv_circuit.measure(i, i)

    return bv_circuit, qiskit.qasm2.dumps(bv_circuit)
