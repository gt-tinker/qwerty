import math
import cmath
import qiskit.qasm2
from qiskit import QuantumCircuit

def get_circuit(n_qubits):
    circ = QuantumCircuit(n_qubits, n_qubits)

    # Use all ones (the last fourier[N] basis vector)
    j = ~(-1 << n_qubits)

    # Equation (5.2) of Nielsen and Chuang
    circ.initialize([cmath.exp(2.*math.pi*1.j*float(j)*float(k)/float(1 << n_qubits))/math.sqrt(float(1 << n_qubits)) for k in range(1 << n_qubits)])
    circ.measure(range(n_qubits), range(n_qubits))
    # reps=6 should be enough in my testing, but let's burn some coal in Macon for fun
    circ = circ.decompose(reps=10)

    return circ, qiskit.qasm2.dumps(circ)
