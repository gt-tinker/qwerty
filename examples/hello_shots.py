from qwerty import *

@qpu
def kernel() -> bit:
    return '0'+'1' | {'0','1'}.measure

histogram(kernel(shots=8192))
