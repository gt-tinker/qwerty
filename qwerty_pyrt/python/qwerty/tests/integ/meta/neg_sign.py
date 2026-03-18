from qwerty import *

@qpu
def neg_sign_func_kernel():
    # '00'+'10' -> '00'-'10' == ('0'-'1')*'0'
    # should yield 10 with 100% probability when measured in pm*std
    return ('0'+'1')*'0' | (-id in '1_') | (pm * std).measure
    #                       ^
    #                       |
    #                       |

# The next kernel is the same as above except -id is defined as a function that
# just returns the negative sign used on its input

@qpu
@reversible
def neg_q(q):
    return -q
    #      ^
    #      |
    #      |

@qpu
def neg_sign_reg_kernel():
    return ('0'+'1')*'0' | (neg_q in '1_') | (pm * std).measure

def test(shots):
    return neg_sign_func_kernel(shots=shots), neg_sign_reg_kernel(shots=shots)
