from qwerty import *

def test(shots):
    @qpu
    def kernel():
        return 'p' | ((pm >> std) > (std >> pm) > (pm >> std) > measure)

    return kernel(shots=shots)
