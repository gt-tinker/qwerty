"""
Implementation of quantum phase estimation (QPE) as defined in Section 5.2 of
Nielsen and Chuang. Takes `prec`[ision], the number of bits of the phase to find;
a function to prepare the eigenvector whose eigenvalue we aim to find
(`get_init_state'); the operator whose eigenvalue to find (`op'); and the number
of shots to run (`shots').
"""

from fractions import Fraction
from qwerty import *

def qpe(prec, get_init_state, op, shots):
    @qpu[[M]]
    def kernel():
        return ('p'**prec * get_init_state()
                | (op[[prec-1-j]]
                   in '?'**j * '1' * '?'**(prec-1-j) * '_'**M
                   for j in range(prec))
                | fourier[prec].measure
                  * discard**M)

    def bits_to_angle_frac(bits):
        return Fraction(int(bits),
                        2**len(bits))

    bits_histo = kernel(shots=shots)
    angle_histo = {bits_to_angle_frac(bits): count
                   for bits, count in bits_histo.items()}
    return angle_histo

@qpu
def init1():
    return '1'

def test(angle_deg, precision, shots):
    @qpu[[J]]
    @reversible
    def tilt_op(q):
        return q | '1' >> '1'@(angle_deg*2**J)

    angle_frac_histo = qpe(precision, init1, tilt_op, shots)
    return {float(360*angle_frac): count
            for angle_frac, count in angle_frac_histo.items()}
