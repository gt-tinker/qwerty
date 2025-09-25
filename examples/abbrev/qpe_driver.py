from qwerty import *
from qpe import qpe

angle_deg = 225.0
precision = 3

@qpu
def init1():
    return '1'

@qpu[[J]]
@reversible
def tilt_op(q):
    return q | '1' >> '1'@(angle_deg*2**J)

print('Expected:', angle_deg)
angle_got = 360*qpe(precision, init1,
                    tilt_op)
print('Actual:', float(angle_got))
