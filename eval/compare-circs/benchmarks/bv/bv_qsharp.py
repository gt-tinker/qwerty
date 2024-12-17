import os
import qsharp
import qiskit.qasm3

src = os.path.join(os.path.dirname(__file__), 'bv.qs')
def get_circuit(qubit_count, secret_string):
    with open(src, 'r') as program:
        qsharp.eval(program.read())
    
    arr_lit = '[' + ','.join(['false', 'true'][int(b)] for b in secret_string) + ']'
    qasm = qsharp.circuit(f'BernsteinVazirani({qubit_count}, PrepareOracle({arr_lit}))').qasm()
    return qiskit.qasm3.loads(qasm), qasm
