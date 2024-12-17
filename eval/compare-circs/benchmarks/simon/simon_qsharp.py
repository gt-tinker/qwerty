import os
import qsharp
import qiskit.qasm3

src = os.path.join(os.path.dirname(__file__), 'simon.qs')
def get_circuit(qubit_count):
    with open(src, 'r') as program:
        qsharp.eval(program.read())

    qasm = qsharp.circuit(f'SimonAlgorithm({qubit_count}, SecretSauceOracle)').qasm()
    return qiskit.qasm3.loads(qasm), qasm
