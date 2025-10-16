from qwerty import *

@qpu
def kernel() -> bit[3]:
    return '0'**3 | ij**3 >> std**2 // {'0', '1'@25}.revolve | std.measure**3

histogram(kernel(shots=1024))
