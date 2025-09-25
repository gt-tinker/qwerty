from qwerty import *
from period import period_finding

@classical
def mod4(x: bit[3]) -> bit[3]:
    return x % 4

period = period_finding(mod4)

if period == 4:
    print('Success!')
else:
    print('Period finding failed')
