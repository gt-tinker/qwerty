"""
Test a basis translation between a 3-qubit basis generated with basis
generators and a 3-fold tensor product of a one-qubit basis. This causes span
equivalence checking to factor the bigger basis.

Because the first vector of ``fourier[N]`` is ``'p'**N``,
``'0'**3 | std**3 >> fourier[3]`` yields ``'p'*3``, and
``'p'**3 | fourier**3 >> std[3]`` yields ``'0'*3``. Thus,
both test cases below should return only bit[3](0b000) measurements.
"""

from qwerty import *

@qpu
def qft() -> bit[3]:
    return '0'**3 | std**3 >> fourier[3] | pm.measure**3

@qpu
def iqft() -> bit[3]:
    return 'p'**3 | fourier[3] >> std**3 | measure**3

def test(shots):
    return (qft(shots=shots),
            iqft(shots=shots))
