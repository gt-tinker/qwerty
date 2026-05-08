from qwerty import *

def test(shots):
    @qpu
    def kernel():
        return '0p' | ((pm >> std) > (std >> pm) > (pm >> std) in '0_') | measure**2

    return kernel(shots=shots)
