from qwerty import *

@qpu_prelude
def standard_prelude():
    # Qubit literals
    '?'.sym = __SYM_PAD__()
    '_'.sym = __SYM_PASSTHRU__()
    '0'.sym = __SYM_STD0__()
    '1'.sym = __SYM_STD1__()
    'p'.sym = '0' or '1'
    'i'.sym = '0' or '1'@90
    'm'.sym = '0' or '1'@180
    'j'.sym = '0' or '1'@270

    # Simple bases
    std = {'0', '1'}
    pm = {'p', 'm'}
    ij = {'i', 'j'}

    # Basis macros
    {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    {bv1, bv2}.rotate = lambda theta: {bv1, bv2} >> {bv1 @ (-theta/2), bv2 @ (theta/2)}
    {bv}.prep = __PREP__(bv)
    b.measure = __MEASURE__(b)
    b.project = __PROJECT__(b)
    b.projmeas = __PROJMEAS__(b)
    b.ticks = __TICKS__(b)

    # Bit macros
    bit[N](val).prep = __PREP_BITS__(N, val)
    bit[N](val).q = '0'[N] | bit[N](val).prep

    # @classical function macros
    f.sign = __SIGN_EMBED__(f)
    f.xor = __XOR_EMBED__(f)
    (f, f_inv).inplace = __INPLACE_EMBED__(f, f_inv)

    # Built-in functions
    id = lambda q: q
    discard = __DISCARD__()
    discardz = __DISCARDZ__()
    flip = std.flip
    measure = std.measure

    # Advanced bases
    fourier[N] = fourier[N-1] ++ std.ticks

@qpu[[N]]
def kernel() -> bit[N]:
    return 'p'[N] | fourier[N].measure

print(histogram(kernel(shots=1024)))
