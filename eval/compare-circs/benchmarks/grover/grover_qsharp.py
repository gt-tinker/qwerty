import os
import qsharp
import qiskit.qasm3

src = os.path.join(os.path.dirname(__file__), 'grover.qs')
def get_circuit(qubit_count, n_iter):
    with open(src, 'r') as program:
        qsharp.eval(program.read())

    qasm = qsharp.circuit(f'Grover({qubit_count}, {n_iter}, Oracle)').qasm()
    return qiskit.qasm3.loads(qasm), qasm
