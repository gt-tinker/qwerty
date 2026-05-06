from qwerty import *

@qpu
def kernel() -> bit:
    f = std >> pm
    return '0' | f | ~f | measure

def test(shots):
    return kernel(shots=shots)
