import os
import qsharp
import qiskit.qasm3

src = os.path.join(os.path.dirname(__file__), 'dj.qs')
def get_circuit(qubit, case):
    function = ''
    if case == 0:
        function = 'ConstantOne'
    elif case >= 1:
        function = 'BalancedEqual'

    with open(src, 'r') as program:
        qsharp.eval(program.read())
    
    qasm = qsharp.circuit(f'DeutschJozsaAlgorithm({qubit}, {function})').qasm()
    return qiskit.qasm3.loads(qasm), qasm