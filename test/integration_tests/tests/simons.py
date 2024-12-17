import sys
import numpy as np
from qwerty import *

class Retry(Exception):
    pass

# The classical post-processing is heavily inspired by a post by Tristan
# Nemoz[1]. However, it has been re-implemented from scratch for licensing
# reasons and to make it easier to read for those who do not know numpy as
# well.
# [1]: https://quantumcomputing.stackexchange.com/a/29407/13156)

# Convert the matrix passed in into row echelon form (REF)
# (assumes all equations in the matrix are mod 2).
# Based on steps 1-4 of the Row Reduction Algorithm in Chapter 1 of Lay, Lay,
# and McDonald.
def ref(mat):
    num_rows, n = mat.shape
    if num_rows != n-1:
        raise ValueError('wrong dimensions for system of equations. there '
                         'should be n-1 equations and n variables')

    # Forward phase: produce row echelon form
    for row in range(num_rows):
        rows_of_nonzeros = mat[row:,row].nonzero()[0]+row

        if rows_of_nonzeros.size:
            # Make sure the first row has a 1 in this column, interchanging rows if
            # necessary
            first_nonzero_row_idx = rows_of_nonzeros[0]
            if first_nonzero_row_idx != row:
                # Swap first nonzero row and first row
                mat[(row, first_nonzero_row_idx),:] = mat[(first_nonzero_row_idx, row),:]

            for nonzero_row_idx in rows_of_nonzeros[1:]:
                # Add the first row (mod 2) to other rows with nonzeros. This will
                # clear all entries in this column below this one
                mat[nonzero_row_idx,:] ^= mat[row,:]
                if not mat[nonzero_row_idx,:].any():
                    raise Retry('linearly dependent')

    return mat

# Converts mat from row echelon form into reduced row echelon form (RREF)
# (assumes all equations in the matrix are mod 2).
# Based on step 5 of the Row Reduction Algorithm in Chapter 1 of Lay, Lay,
# and McDonald.
def rref(mat):
    num_rows, n = mat.shape

    # Backward phase: produce _reduced_ row echelon form
    for row in reversed(range(num_rows)):
        cols_of_nonzeros = mat[row,:].nonzero()[0]

        if not cols_of_nonzeros.size:
            # Should have been caught in ref(), but check just in case
            raise Retry('linearly dependent')

        pivot_col_idx = cols_of_nonzeros[0]
        nonzero_rows_above = mat[:row,pivot_col_idx].nonzero()[0]

        for nonzero_row_idx in nonzero_rows_above:
            # Add this row (mod 2) to earlier rows with nonzeros in this
            # column. This will clear all entries in this column above this one
            mat[nonzero_row_idx,:] ^= mat[row,:]
            if not mat[nonzero_row_idx,:].any():
                raise Retry('linearly dependent')

    return mat

# Determines the secret string from a system of linearly independent equations
# expressed as a matrix in reduced row echelon form with no rows of all zeros.
def extract_secret_string(mat):
    num_rows, n = mat.shape

    for i in range(n):
        if i < num_rows:
            if not mat[i,i]:
                special_col_idx = i
                break
        else:
            special_col_idx = n-1

    s = ''.join(str(b) for b in mat[:special_col_idx,special_col_idx])
    s += '1'
    s += '0'*(n-special_col_idx-1)
    return bit.from_str(s)

def simons(f):
    @qpu[[N]](f)
    def kernel(f: cfunc[N]) -> bit[N]:
        return 'p'[N] + '0'[N] | f.xor \
                               | (pm[N] >> std[N]) + discard[N] \
                               | std[N].measure

    while True:
        rows = []
        while True:
            meas = kernel()
            if meas: # Ignore unhelpful zero measurements
                rows.append(list(meas.get_bits()))
                if len(rows) >= meas.n_bits-1:
                    break

        mat = np.asarray(rows)
        try:
            secret_string = extract_secret_string(rref(ref(mat)))
        except Retry:
            print('retrying (this is expected)... ', file=sys.stderr, end='', flush=True)
            continue

        # success!
        return secret_string

# This black box was chosen by writing a truth table and then converting it to
# classical logic (specifically using a K-Map)
@classical
def black_box(q: bit[3]) -> bit[3]:
    return (~q[0] & q[2] | q[0] & ~q[2] | ~q[1]), \
           (~q[0] & ~q[2] | q[0] & q[2]), \
           (~q[0] & ~q[2] | q[0] & q[2] | ~q[1])

def test():
    return simons(black_box)
