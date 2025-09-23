from qwerty import *

def bv(secret_str):
    @classical[[N]]
    def oracle(x: bit[N]) -> bit:
        return (secret_str & x).xor_reduce()

    @qpu[[N]]
    def kernel():
        return ('p'**N | oracle.sign
                       | pm**N >> std**N
                       | measure**N)

    return kernel()
