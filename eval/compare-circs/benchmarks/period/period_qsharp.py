import os
import qsharp
import qiskit.qasm3

src = os.path.join(os.path.dirname(__file__), 'period.qs')
def get_circuit(n_bits_in, n_bits_out, n_mask_bits):
    with open(src, 'r') as program:
        qsharp.eval(program.read())

    qasm = qsharp.circuit(f'PeriodFinding({n_bits_in}, {n_bits_out}, PrepareOracle({n_mask_bits}))').qasm()
    return qiskit.qasm3.loads(qasm), qasm
