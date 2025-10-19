import os
import cmath
import qsharp
import qiskit.qasm3

from .vectors import vecs

src = os.path.join(os.path.dirname(__file__), 'rand.qs')
def get_circuit(qubit_count):
    with open(src, 'r') as program:
        qsharp.eval(program.read())

    vec = vecs[qubit_count]
    array = '[' + ','.join('ComplexPolar({}, {})'.format(abs(v), cmath.phase(v)) for v in vec) + ']'
    qasm = qsharp.circuit(f'RandPrep({qubit_count}, {array})').qasm()
    return qiskit.qasm3.loads(qasm), qasm
