"""
Prepares the Fourier basis state from Fig. 12 of the QCE '25 paper and measures
in the Fourier basis, all without metaQwerty features.
"""

from qwerty import *

@qpu(prelude=None)
def kernel() -> bit[3]:
    return ((__SYM_STD0__() + __SYM_STD1__()@180) * (__SYM_STD0__() + __SYM_STD1__()@90) * (__SYM_STD0__() + __SYM_STD1__()@225)
            | __MEASURE__(({__SYM_STD0__()+__SYM_STD1__(), __SYM_STD0__()-__SYM_STD1__()} // __REVOLVE__(__SYM_STD0__(),__SYM_STD1__())) // __REVOLVE__(__SYM_STD0__(),__SYM_STD1__())))

kernel(shots=1024)
