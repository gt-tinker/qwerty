"""
A version of Bernstein–Vazirani with neither metaQwerty features nor classical
function embeddings.
"""

from qwerty import *

@qpu(prelude=None)
def kernel() -> bit[3]:
    # secret string is 110
    f_sign = {__SYM_STD0__()*__SYM_STD1__()*__SYM_STD0__(), __SYM_STD0__()*__SYM_STD1__()*__SYM_STD1__(), __SYM_STD1__()*__SYM_STD0__()*__SYM_STD0__(), __SYM_STD1__()*__SYM_STD0__()*__SYM_STD1__()} >> {-__SYM_STD0__()*__SYM_STD1__()*__SYM_STD0__(), -__SYM_STD0__()*__SYM_STD1__()*__SYM_STD1__(), -__SYM_STD1__()*__SYM_STD0__()*__SYM_STD0__(), -__SYM_STD1__()*__SYM_STD0__()*__SYM_STD1__()}
    return ((__SYM_STD0__()+__SYM_STD1__())*(__SYM_STD0__()+__SYM_STD1__())*(__SYM_STD0__()+__SYM_STD1__())
            | f_sign
            | {__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()}*{__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()}*{__SYM_STD0__()+__SYM_STD1__(),__SYM_STD0__()-__SYM_STD1__()} >> {__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}
            | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}))

def test(shots):
    return kernel(shots=shots)
