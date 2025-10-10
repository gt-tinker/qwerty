"""
Run QPE in QIR-EE. This is pretty much ``examples/qpe.py``.
"""

from fractions import Fraction
from argparse import ArgumentParser
from qwerty import *

def qpe(prec, get_init_state, op, shots, acc=None):
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

    bits_histo = kernel(shots=shots, acc=acc)
    angle_histo = {bits_to_angle_frac(bits): count
                   for bits, count in bits_histo.items()}
    return angle_histo

def test(angle_deg, precision, shots, acc=None):
    @qpu
    def init1():
        return '1'

    @qpu[[J]]
    @reversible
    def tilt_op(q):
        return q | '1' >> '1'@(angle_deg*2**J)

    angle_frac_histo = qpe(precision, init1, tilt_op, shots, acc=acc)
    return {float(360*angle_frac): count
            for angle_frac, count in angle_frac_histo.items()}
