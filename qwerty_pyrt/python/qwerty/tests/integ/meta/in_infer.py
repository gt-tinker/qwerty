from qwerty import *

def test(shots):
    @qpu
    def kernel():
        return ('p' * '1000'
                # metaQwerty expansion introduces dimension variables, which
                # means that dimension variable inference needs to be good
                # enough for this to work
                | ('1???'>>-'1???' in '1'*'_'**4)
                | pm.measure * discard**4)

    return kernel(shots=shots)
