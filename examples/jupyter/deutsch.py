"""
This is a standalone Qwerty equivalent of the code in ``demo.ipynb``.
"""

from qwerty import *

@classical
def f(x: bit) -> bit:
    return ~x

@qpu
def deutsch() -> bit:
    flip = {'0'>>'1', '1'>>'0'}
    flop = {'0'+'1'>>'0', '0'-'1'>>'1'}
    return '0'+'1' | f.sign | flop | measure

print(deutsch())