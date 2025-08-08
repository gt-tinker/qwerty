from qwerty import *

@qpu
def kernel() -> bit[3]:
    q1, q2, q3 = (__SYM_STD0__()-__SYM_STD1__())*__SYM_STD0__()*__SYM_STD1__()
    q1m = __MEASURE__({__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()})(q1)
    q23 = q2*q3
    q2m, q3m = __MEASURE__({__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()})(q23)
    return q1m * q2m * q3m

def test(shots):
    return kernel(shots=shots)
