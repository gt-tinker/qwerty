#!/usr/bin/env python3

import sys
from qwerty import *
from random import randint

def quantum_walk(steps, shots):
    @qpu
    def walk_step(q):
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
                  | (plus_one if '1__' else id**2)
                  | (minus_one if '0__' else id**2))

    @qpu
    def kernel():
        return ('i00' | (walk_step for i in range(steps))
                      | discard * measure**2)

    print('Quantum walk:')
    histogram(kernel(shots=shots))

def classical_walk(steps, shots):
    histo = {}
    for _ in range(shots):
        state = 0b00
        for i in range(steps):
            coin_flip = randint(0, 1)
            state = (state + coin_flip) & 0b11
        state_bits = bit[2](state)
        histo[state_bits] = histo.get(state_bits, 0) + 1

    print('Classical walk:')
    histogram(histo)

def main(args):
    steps = int(args[0]) if args else 100
    shots = int(args[1]) if len(args) > 1 else 1024

    quantum_walk(steps, shots)
    print()
    classical_walk(steps, shots)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
