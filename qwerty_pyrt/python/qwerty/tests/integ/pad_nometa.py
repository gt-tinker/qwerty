from qwerty import *

@qpu
def kernel() -> bit[3]:
    q = '001' | '???' >> '???' | {'0?1','1?0'} >> {'1?0','0?1'} | {'?0?', '?1?'} >> {'?1?', '?0?'}
    return q | __MEASURE__({'0','1'}*{'0','1'}*{'0','1'})

def test(shots):
    return kernel(shots=shots)
