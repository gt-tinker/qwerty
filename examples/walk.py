#!/usr/bin/env python3

from qwerty import *

steps = 5

@qpu
def walk_step(q):
    '↑'.sym = '0'
    '↓'.sym = '1'

    coin = std * '??' >> pm * '??'
    plus_one = {'00' >> '01',
                '01' >> '10',
                '10' >> '11',
                '11' >> '00'}
    minus_one = {'00' >> '11',
                 '01' >> '00',
                 '10' >> '01',
                 '11' >> '10'}

    return (q | coin
              | (plus_one if '↑__' else id**2)
              | (minus_one if '↓__' else id**2))

@qpu
def kernel():
    '↑'.sym = '0'
    '↓'.sym = '1'

    return ('↑'*'00' | (walk_step for i in range(steps))
                     | discard * measure**2)

histogram(kernel(shots=20480))
