from qwerty import *

@qpu
def kernel() -> bit:
    return '0' | pm.rotate(pi/2) | ~pm.rotate(pi/2) | std.measure

def test(n_shots):
    return kernel(shots=n_shots, histogram=True)
