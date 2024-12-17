from qwerty import *

@qpu
def f(q: qubit) -> bit:
    return (q | std >> {'1','0'} if ('p' | std.measure) else q | id) | std.measure

@qpu(f)
def kernel(f: func[[qubit],bit]) -> bit:
    return '0' | f

def test():
    return kernel()
