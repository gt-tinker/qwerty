from qwerty import *
from superdense import superdense_coding

for i in range(1 << 2):
    payload = bit[2](i)
    print('{} -> {}'.format(
        payload, superdense_coding(payload)))
