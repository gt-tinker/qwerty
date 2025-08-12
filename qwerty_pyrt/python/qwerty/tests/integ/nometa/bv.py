"""
A version of Bernstein–Vazirani with no metaQwerty features.
"""

from qwerty import *

secret_string = bit[3](0b110)

@classical(prelude=None)
def oracle(x: bit[3]) -> bit:
    return (x & secret_string).xor_reduce()

@qpu(prelude=None)
def kernel() -> bit[3]:
    return ((__SYM_STD0__()+__SYM_STD1__())*(__SYM_STD0__()+__SYM_STD1__())*(__SYM_STD0__()+__SYM_STD1__())
            | __EMBED_SIGN__(oracle)
            | {__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()}*{__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()}*{__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()} >> {__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}
            | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}))

def test(shots):
    return kernel(shots=shots)
