from qwerty import *

@qpu
def kernel() -> bit[3]:
    return ('mi' * ('0' + '1'@225)
            | fourier[3].measure)

kernel(shots=1024)
