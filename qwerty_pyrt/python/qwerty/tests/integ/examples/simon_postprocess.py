"""
The classical post-processing for Simon's (simons.py) is lengthy enough that we
write it here in its own file. This module is not intended to be run on its own.

This classical post-processing is heavily inspired by
`a post by Tristan Nemoz`_. However, it has been re-implemented from scratch
for licensing reasons and to make it easier to read for those who do not know
numpy as
well.

.. _a post by Tristan Nemoz: https://quantumcomputing.stackexchange.com/a/29407/13156
"""

import numpy as np
from qwerty import *

class Retry(Exception):
    pass


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

    # (In the following discussion, indices are assumed to begin at 1, not 0.
    #  Note this differs from numpy, where indices begin at 0.)
    #
    # Definition 1: A **n-fold Simon matrix** is a (n-1)xn matrix in reduced
    #               row echelon form containing only 0 and 1 entries and
    #               without any rows of all zeros.
    #
    # The input to this function (mat) is a Simon matrix. The n variables in
    # the system of equations are exactly the n bits of a secret string s,
    # which cannot consist of all 0s.
    #
    # Definition 2: The **special column** of a n-fold Simon matrix M is the
    #               column whose index is the smallest 1 <= c <= n such that
    #               either c = n or M[c,c] = 0.
    #
    # Lemma 3: Suppose M is an n-fold Simon matrix whose special column has
    #          index c. Then if c <= i < n and 1 <= j <= n,
    #                   { 1  if j = i+1
    #          M[i,j] = {
    #                   { 0  otherwise
    # Proof:
    # Suppose c < n, since otherwise the statement of the lemma is not
    # applicable. By definition of special column, M[k,k] = 1 for all
    # 1 <= k < c, yet M[c,c] = 0. By definition of Simon matrix, however, the
    # row at index c cannot consist of all zeros. Thus, there must be an entry
    # M[c,l] = 1 with c < l <= n-1. Yet for each c < j <= n-1, the row with
    # index j must also have a nonzero by definition of Simon matrix. In order
    # not to violate the definition of reduced row echelon form, then, l=c+1
    # and M[j,m] = 1 if m=j+1 and 0 otherwise for all j. Qed.
    #
    #
    # Lemma 4: Suppose M is an n-fold Simon matrix whose special column has
    #          index c. Then every row 1 <= i < c consists of zeros except for
    #          a leading entry in column i and possibly a 1 in column c.
    # Proof:
    # By definition of special column, M[i,i] = 1 for all i, since i < c. Then
    # for all 1 <= k_i < i, M[i, k_i] = 0 by definition of row echelon form, and
    # for all i < l_i < c, M[i, l_i] = 0 by definition of reduced row echelon
    # form. And by Lemma 3 and the definition of reduced row echelon form, for
    # all c < h_i <= n, M[i, h_i] = 0. Thus, M[i,c] is the only possible other
    # 1 in row i besides the leading entry M[i,i]. Qed.
    #
    #
    # Theorem: If M is an n-fold Simon matrix and its special column is at
    #          index c, then the n-bit secret string is
    #          M[1,c],M[2,c],...,M[c-1,c],1,0,0,...,0.
    # Proof:
    # Lemma 3 implies there are equations s_i' = 0 (mod 2) with c < i' <= n.
    # Thus, the last n-c bits of the secret string are 0.
    #
    # By Lemma 4, each row at index i' such that 1 <= i < c represents an
    # equation s_i + M[i,c]*s_c = 0 (mod 2).
    #
    # Now, for the purpose of contradiction, assume that s_c = 0. If s_c = 0,
    # though, each aforementioned equation becomes s_i = 0 (mod 2), so the
    # first c-1 bits of the secret string are 0. However, if the first c-1 bits
    # of the secret string are 0, and bit c of the secret string is zero (since
    # s_c=0), and the last n-c bits of the secret string are 0, then the secret
    # string s must consist of all 0s. This contradicts the assumption that s
    # does not consist of all zeros, so we must conclude that s_c = 1.
    #
    # Consequently, each aforementioned equation is equivalent to
    # s_i + M[i,c] = 0 (mod 2), which is equivalent to s_i = M[i,c] (mod 2). In
    # other words, the first c-1 bits of the secret string are the first c-1
    # elements of column c of M. The next bit is 1 (since we proved that
    # s_c=1), and the remaining n-c bits are zeros as initially shown. Qed.

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

def simon_post(rows):
    matrix = np.asarray([list(meas.get_bits()) for meas in rows])
    return extract_secret_string(rref(ref(matrix)))
