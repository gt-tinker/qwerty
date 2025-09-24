from qwerty import *
from deutsch import deutsch

@classical
def balanced(x: bit) -> bit:
    return ~x

@classical
def constant(x: bit) -> bit:
    return bit[1](0b1)

print('If f(x) = ~x:')
print('f(0) xor f(1) =', deutsch(balanced))
print('If f(x) = 1:')
print('f(0) xor f(1) =', deutsch(constant))
