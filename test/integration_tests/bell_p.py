from qwerty import *

@qpu
def cnot(q: qubit[2]) -> qubit[2]:
     return q | '1' + std >> '1' + {'1', '0'}

@qpu(cnot)
def bell(cnot: qfunc[2]) -> qubit[2]:
    return '+0' | cnot

@qpu(bell)
def kernel(bell: func[[], qubit[2]]) -> bit[2]:
    return bell() | std[2].measure


def test():
    return kernel()
