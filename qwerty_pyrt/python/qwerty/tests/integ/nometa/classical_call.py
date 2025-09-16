from qwerty import *

def test(secret_string):
    @classical
    def oracle(x: bit[3]) -> bit:
        return (x & secret_string).xor_reduce()

    return [oracle(x) for x in range(1 << 3)]
