from qwerty import *

@qpu
def kernel() -> bit[3]:
    q1, q2, q3 = 'm01'
    q1m = pm.measure(q1)
    q23 = q2 * q3
    q2m, q3m = (measure * measure)(q23)
    return q1m * q2m * q3m

def test(shots):
    return kernel(shots=shots)
