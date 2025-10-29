import os
import qsharp
import qiskit.qasm3

src = os.path.join(os.path.dirname(__file__), 'qft.qs')
def get_circuit(qubit_count):
    with open(src, 'r') as program:
        qsharp.eval(program.read())

    qasm = qsharp.circuit(f'QFT({qubit_count})').qasm()
    return qiskit.qasm3.loads(qasm), qasm
