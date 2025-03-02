from qwerty import *

# Test syntax sugar

@qpu
def sweet() -> bit:
    return 'p' | {'p' >> '0', 'm' >> '1'} | measure
