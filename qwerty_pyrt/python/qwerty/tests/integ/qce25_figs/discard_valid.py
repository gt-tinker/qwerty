"""
Fig. 4(b) of the Qwerty QCE '25 paper.
"""

from qwerty import *

@qpu
def valid():
    a, b = '01' + '10'
    return a * b | measure * discard

def test(shots):
    return valid(shots=shots)
