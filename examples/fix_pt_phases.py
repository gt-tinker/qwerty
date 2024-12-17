"""
Generating the phases for fixed-point amplitude amplification (fix_pt_amp.py)
is complex enough to relegate it to its own file. The phase generation code is
taken from pyqsp [1].

This module is not intended to be run on its own.

[1] J. M. Martyn, Z. M. Rossi, A. K. Tan, and I. L. Chuang, “Grand
    Unification of Quantum Algorithms,” PRX Quantum, vol. 2, no. 4, p. 040203,
    Dec. 2021, doi: 10.1103/PRXQuantum.2.040203.
"""

import math
import numpy as np

# Taken directly from pyqsp
# https://github.com/ichuang/pyqsp/blob/6ec2499c153b4359a11c914ff59bff16545508cc/pyqsp/phases.py#L51-L77
def generate_Wx_fix_pt_phases(d, delta):
    L = 2 * d + 1
    # Generates length {2*d} sequence in the Wx convention")
    kvec = np.arange(1, d + 1)
    # T_{1/L}(1/delta)
    gamma = 1 / np.cosh((1 / L) * np.arccosh(1 / delta))
    sg = np.sqrt(1 - gamma**2)
    avec = 2 * np.arctan2(1, (np.tan(2 * np.pi * kvec / L) * sg))
    bvec = - avec[::-1]
    phivec = [0.0]*(2*d)
    for k in range(d):
        # reverse order & scale to match QSVT convention
        phivec[2 * k] = -avec[d - k - 1] / 2
        phivec[2 * k + 1] = bvec[d - k - 1] / 2

    return phivec

def get_phases(original_prob, desired_prob):
    delta = math.sqrt(1-desired_prob)
    L_lower_bound = math.ceil(math.log2(2/delta)/math.sqrt(original_prob))
    # Want an odd L by definition of little l
    L = L_lower_bound + 1 - (L_lower_bound % 2)
    l = (L - 1)//2

    phases = generate_Wx_fix_pt_phases(l, delta)
    # In Qwerty, you write operations in the opposite order as matrix
    # multiplication, so make the code below easier to understand by reversing
    # the order of QSP phases vs. the Martyn/Gilyen papers.
    # We also scale up by 2 to account for using Rz(theta) below versus
    # exp(-i*theta*Z) Martyn and Gilyen use.
    phases = [2*p for p in reversed(phases)]
    return phases
