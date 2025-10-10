from qwerty import *

@qpu
def kernel() -> bit[3]:
    return '0'**3 | std**3 >> std**2 // std.revolve | std.measure**3

histogram(kernel(shots=1024))
