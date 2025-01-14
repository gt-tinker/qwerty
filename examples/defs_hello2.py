from qwerty import *

@typedefs
def standard_typedefs():
    # Type annotations (registers)
    qubit = __TYPE_REG_QUBIT__(1)
    qubit[N] = __TYPE_REG_QUBIT__(N)

    # Type annotations (quantum functions)
    func[[*arg_types], ret_type].type = __TYPE_FUNC__(ret_type, *arg_types)
    rev_func[[*arg_types], ret_type].type = __TYPE_REV_FUNC__(ret_type, *arg_types)
    qfunc.type = func[[qubit], qubit]
    qfunc[N].type = func[[qubit[N]], qubit[N]]
    qfunc[M, N].type = func[[qubit[M]], qubit[N]]
    rev_qfunc.type = rev_func[[qubit], qubit]
    rev_qfunc[N].type = rev_func[[qubit[N]], qubit[N]]
    rev_qfunc[M, N].type = rev_func[[qubit[M]], qubit[N]]

    # Type annotations (classical functions)
    bit = __TYPE_REG_BIT__(1)
    bit[N] = __TYPE_REG_BIT__(N)
    cfunc.type = __TYPE_CLASSICAL_FUNC__(bit, bit)
    cfunc[N].type = __TYPE_CLASSICAL_FUNC__(bit[N], bit[N])
    cfunc[M, N].type = __TYPE_CLASSICAL_FUNC__(bit[M], bit[N])

@qpu_prelude
def standard_qpu_prelude():
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
    std[1] = {'0', '1'}
    pm[1] = {'p', 'm'}
    ij[1] = {'i', 'j'}

    # Basis macros
    {bv1, bv2}.rev = {bv2, bv1}
    b.flip = b >> b.rev
    {bv1, bv2}.rotate = lambda theta: {bv1, bv2} >> {bv1 @ (-theta/2), bv2 @ (theta/2)}
    {bv}.prep = __PREP__(bv)
    b.measure = __MEASURE__(b)
    b.project = __PROJECT__(b)
    b.projmeas = __PROJMEAS__(b)
    {bv1, bv2}.spin = __SPIN__(bv1, bv2)

    # Bit macros
    bit[N](val).prep = __PREP_BITS__(N, val)
    bit[N](val).q = '0'[N] | bit[N](val).prep

    # @classical function macros
    f.sign = __SIGN_EMBED__(f)
    f.xor = __XOR_EMBED__(f)
    (f, f_inv).inplace = __INPLACE_EMBED__(f, f_inv)

    # Built-in functions
    id: qfunc = lambda q: q
    discard = __DISCARD__()
    discardz = __DISCARDZ__()
    flip = std.flip
    measure = std.measure

    # Advanced bases
    fourier[N] = fourier[N-1] ++ std.spin

@qpu[[N]]
def kernel() -> bit[N]:
    return 'p'[N] | fourier[N].measure

print(histogram(kernel(shots=1024)))
