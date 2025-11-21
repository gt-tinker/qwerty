import sys
from qwerty import *

N = int(sys.argv[1])

@qpu
def kernel():
    return 'p'**N | fourier[N].measure

with open('iqft.qasm', 'w') as fp:
    fp.write(kernel.qasm())
