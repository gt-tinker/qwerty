"""
Fig. 3 of the Qwerty QCE '25 paper represented as a full Qwerty program.
"""

from qwerty import *

@qpu
def superpos() -> bit:
    return 'p' + 'm' | measure

@qpu
def ensemble() -> bit:
    return 'p' ^ 'm' | measure

def test(shots):
    return superpos(shots=shots), ensemble(shots=shots)
