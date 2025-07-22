from qwerty import *

@qpu
def kernel() -> bit[3]:
    q1, q2, q3 = ('0'-'1')*'01'
    q1m = __MEASURE__({'0'+'1','0'-'1'})(q1)
    q23 = q2*q3
    q2m, q3m = __MEASURE__({'0','1'}*{'0','1'})(q23)
    return q1m * q2m * q3m

def test(shots):
    return kernel(shots=shots)
