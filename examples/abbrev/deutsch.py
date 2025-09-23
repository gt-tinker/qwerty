from qwerty import *

def deutsch(oracle):
    @qpu
    def kernel():
        return 'p' | oracle.sign | pm.measure

    return kernel()
