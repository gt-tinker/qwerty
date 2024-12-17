from qwerty import *

@qpu
def kernel() -> bit:
    return '0' | () + std >> {'1','0'} + std[0] | std.measure

def test(n_shots):
    return kernel(shots=n_shots, histogram=True)
