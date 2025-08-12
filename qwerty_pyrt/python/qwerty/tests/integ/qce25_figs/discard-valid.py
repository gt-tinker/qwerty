"""
Fig. 4(b) of the Qwerty QCE '25 paper.
"""

@qpu
def valid():
    a, b = '01' + '10'
    return a * b | measure * discard
