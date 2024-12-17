from qwerty import *

@qpu
def kernel() -> bit[66]:
    return '10'[32] + '10' | '10'[32] + '1' + std >> '10'[32] + '1' + {'1', '0'} | std[66].measure

def test():
    return kernel()
