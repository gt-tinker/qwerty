"""
Deutsch–Jozsa. Included as an integration test to make sure that `x == y` in
`@classical` functions is tested, including when executed classically.
"""

from qwerty import *

def deutsch_jozsa(f, shots=None):
    @qpu[[N]]
    def kernel():
        return ('p'**N | f.sign
                       | pm.measure**N)

    return kernel(shots=shots)

@classical
def constant(x: bit[4]) -> bit:
    # f(x) = 1
    return bit[1](0b1)

@classical
def balanced(x: bit[4]) -> bit:
    # f(x) = 1 for half the inputs
    # and f(x) = 0 for the other half
    return x[0] == x[3]

def naive_classical(f, n_bits):
    answers = [0, 0]
    for i in range(2**(n_bits-1)+1):
        answer = int(f(bit[n_bits](i)))
        answers[answer] += 1

    if 0 in answers:
        return 'constant'
    else:
        return 'balanced'

def test(shots):
    return [(naive_classical(constant, 4),
             deutsch_jozsa(constant, shots=shots)),
            (naive_classical(balanced, 4),
             deutsch_jozsa(balanced, shots=shots))]
