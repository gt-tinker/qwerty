from qwerty import *

# Proof-of-concept for language syntax sugar.

@qpu
def sweet() -> bit:
    # return 'p' | {'p', 'm'} >> {'0', '1'} | measure
    return 'p' | {'p' >> '0', 'm' >> '1'} | measure
    # All of the following should cause typechecking to fail: 
    # return '0' | {{'0', '1'} >> {'p', 'm'}, {'p', 'm'} >> {'i', 'j'}} | measure
    # return '0' | {{'0', '1'} >> {'p', 'm'}, 'p' >> 'i'} | measure

print(sweet())
