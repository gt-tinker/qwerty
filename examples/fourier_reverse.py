from qwerty import *

@qpu
def kernel() -> bit[3]:
    return '0'**3 | fourier[3] >> std**3 | std.measure**3

kernel(shots=1024)
