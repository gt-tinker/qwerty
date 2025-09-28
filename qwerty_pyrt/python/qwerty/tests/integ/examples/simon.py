"""
A Qwerty implementation of Simon's algorithm, the first quantum algorithm to
promise exponential speedup.

When run directly, this module also acts as a tester for the Qwerty
implementation of Simon's. The number of qubits passed on the command line and
the secret string is printed out. The black box is a custom construction to
guarantee a particular 2-to-1 mapping as required by Simon's with a nonzero
secret string.
"""

from qwerty import *

from .simon_postprocess import simon_post, Retry

def simon(f, num_attempts):
    @qpu[[N]]
    def kernel():
        return ('p'**N * '0'**N | f.xor
                                | (std**N >> pm**N) * id**N
                                | measure**N * discard**N)

    for _ in range(num_attempts):
        rows = []
        for _ in range(num_attempts):
            row = kernel()
            if int(row) != 0:
                rows.append(row)
                if len(rows) >= row.n_bits-1:
                    break
        else:
            raise Exception(f'exceeded {num_attempts} tries')

        try:
            return simon_post(rows)
        except Retry:
            print('retrying (this is expected)...')
            continue
    else:
        raise Exception(f'exceeded {num_attempts} tries')

def get_black_box(num_bits):
    if num_bits % 2 != 0:
        raise ValueError(f'Number of bits {num_bits} must be even')

    k = num_bits//2

    @classical
    def black_box(x: bit[2*k]) -> bit[2*k]:
        return x[:k].concat(bit[1](0b0)).concat(x[k].repeat(k-1) ^ x[k+1:])

    return black_box

def naive_classical(f, num_bits):
    out_to_x = {}
    for i in range(2**num_bits):
        x = bit[num_bits](i)
        out = f(x)
        if out in out_to_x:
            return x ^ out_to_x[out]
        out_to_x[out] = x

def test(num_bits, num_attempts):
    f = get_black_box(num_bits)

    return (naive_classical(f, num_bits), simon(f, num_attempts))
