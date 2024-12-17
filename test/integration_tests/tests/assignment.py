from qwerty import *

@qpu
def bubba(q: qubit) -> bit:
    q2 = q | std >> pm
    return q2 | pm.measure

@qpu(bubba)
def kernel(bubba: func[[qubit], bit]) -> bit:
    return '0' | bubba

def test():
    return kernel()

