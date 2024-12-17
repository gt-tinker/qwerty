from qwerty import *

# Based on Section 1.4.4 of Nielsen and Chuang
def deutsch_jozsa(f):
    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] | f.sign         \
                      | pm[N] >> std[N] \
                      | std[N].measure

    if kernel(): # We measured a nonzero bit
        return 'f(x) is balanced'
    else:
        return 'f(x) is constant'

@classical
def constant(x: bit[4]) -> bit:
    # f(x) = 1
    return bit[1](1)

@classical
def balanced(x: bit[4]) -> bit:
    # f(x) = 1 for half the inputs
    # and f(x) = 0 for the other half
    return x.xor_reduce()


def test():
    return deutsch_jozsa(constant) + " | " + deutsch_jozsa(balanced)
