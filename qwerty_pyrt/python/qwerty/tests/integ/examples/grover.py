"""
Implementation of Grover's algorithm as defined in Section 6.1 of Nielsen and
Chuang. Takes three inputs: (1) a classical function `oracle' that returns 1 on
a matching N-bit standard basis state; (2) a number of iterations to run; and
(3) the number of samples to collect (`n_shots').
"""

import math
from argparse import ArgumentParser
from qwerty import *

def grover(oracle, num_iter, shots):
    @qpu[[N]]
    def grover_iter(q):
        return (q | oracle.sign
                  | 'p'**N >> -'p'**N)

    @qpu[[N]]
    def kernel():
        return ('p'**N | (grover_iter for i in range(num_iter))
                       | measure**N)

    results = kernel(shots=shots)
    return (results, list(sorted(x for x in set(results) if oracle(x))))

def calc_num_iter(num_qubits, num_answers):
    n = 2**num_qubits
    m = num_answers
    theta = 2*math.acos(math.sqrt((n-m)/n))
    rnd = lambda x: math.ceil(x-0.5)
    return rnd(math.acos(math.sqrt(m/n))/theta)

def get_black_box(num_qubits):
    @classical
    def all_ones(x: bit[num_qubits]) -> bit:
        return x.and_reduce()

    return all_ones

def test(num_qubits, shots):
    num_answers = 1
    oracle = get_black_box(num_qubits)
    num_iter = calc_num_iter(num_qubits, num_answers)
    return grover(oracle, num_iter, shots)
