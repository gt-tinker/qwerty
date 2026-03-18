#!/usr/bin/env python3

import sys
import random
from qwerty import *

def quantum_walk(steps, shots):
    @qpu
    def walk_step(q):
        coin = -('pp' >> -'pp')
        return (q | coin * id**4
                  | (flip in '00_???')
                  | (flip in '01?_??')
                  | (flip in '10??_?')
                  | (flip in '11???_'))

    @qpu
    def kernel():
        return ('00'*'0000' | (walk_step for i in range(steps))
                            | discard**2 * measure**4)

    print('Quantum walk:')
    histogram(kernel(shots=shots))

def classical_walk(steps, shots):
    histo = {}
    for _ in range(shots):
        state = 0b0000
        for i in range(steps):
            coin_flip = random.randint(0, 3)
            state ^= 1 << (3 - coin_flip)
        state = bit[4](state)
        histo[state] = histo.get(state, 0) + 1

    print('Classical walk:')
    histogram(histo)

def main(args):
    steps = int(args[0]) if args else 64
    shots = int(args[1]) if len(args) > 1 else 1024

    quantum_walk(steps, shots)
    print()
    classical_walk(steps, shots)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
