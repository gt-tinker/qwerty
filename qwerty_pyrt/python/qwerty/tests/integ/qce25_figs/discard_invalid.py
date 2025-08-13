"""
Fig. 4(a) of the Qwerty QCE '25 paper.
"""

from qwerty import *

@qpu
def invalid():
    a, b = '01' + '10'
    return a | measure

def test():
    return invalid()
