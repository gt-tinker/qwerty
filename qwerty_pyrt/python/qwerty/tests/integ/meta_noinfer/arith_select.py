from qwerty import *

@qpu
def kernel() -> bit:
    r0 = '0'|measure
    r1 = '1'|measure
    return  r0 if 'p'|measure else r1

def test():
    return kernel()
