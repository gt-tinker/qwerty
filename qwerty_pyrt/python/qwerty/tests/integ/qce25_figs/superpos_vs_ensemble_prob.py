"""
Fig. 3 of the Qwerty QCE '25 paper represented as a full Qwerty program, except
with the probability syntax from Section III-B.
"""

from qwerty import *

@qpu
def superpos() -> bit:
    return 0.5*'p' + 0.5*'m' | measure

@qpu
def ensemble() -> bit:
    return 0.5*'p' ^ 0.5*'m' | measure

def test(shots):
    return superpos(shots=shots), ensemble(shots=shots)
