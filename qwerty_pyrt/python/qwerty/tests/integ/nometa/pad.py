from qwerty import *

@qpu(prelude=None)
def kernel() -> bit[3]:
    q = __SYM_STD0__()*__SYM_STD0__()*__SYM_STD1__() | __SYM_PAD__()*__SYM_PAD__()*__SYM_PAD__() >> __SYM_PAD__()*__SYM_PAD__()*__SYM_PAD__() | {__SYM_STD0__()*__SYM_PAD__()*__SYM_STD1__(),__SYM_STD1__()*__SYM_PAD__()*__SYM_STD0__()} >> {__SYM_STD1__()*__SYM_PAD__()*__SYM_STD0__(),__SYM_STD0__()*__SYM_PAD__()*__SYM_STD1__()} | {__SYM_PAD__()*__SYM_STD0__()*__SYM_PAD__(), __SYM_PAD__()*__SYM_STD1__()*__SYM_PAD__()} >> {__SYM_PAD__()*__SYM_STD1__()*__SYM_PAD__(), __SYM_PAD__()*__SYM_STD0__()*__SYM_PAD__()}
    return q | __MEASURE__({__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()}*{__SYM_STD0__(),__SYM_STD1__()})

def test(shots):
    return kernel(shots=shots)
