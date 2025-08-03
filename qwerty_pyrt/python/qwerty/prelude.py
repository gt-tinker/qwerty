"""
The definition of the default Qwerty prelude, which defines common qubit
symbols, bases, macros, and abbreviated operation names.
"""

@qpu_prelude
def default_prelude():
    # Symbols in qubit literals
    '0'.sym = __SYM_STD0__()
    '1'.sym = __SYM_STD1__()
    'p'.sym = '0' + '1'
    'i'.sym = '0' + '1'@90
    'm'.sym = '0' + '1'@180
    'j'.sym = '0' + '1'@270

    # Common bases
    std = {'0', '1'}
    pm = {'p', 'm'}
    ij = {'i', 'j'}
    bell = {'00'+'11', '00'-'11', '10'+'01', '01'-'10'}

    # Basis macros
    {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    b.measure = __MEASURE__(b)
    {bv1, bv2}.revolve = __REVOLVE__(bv1, bv2)

    # More complicated bases
    fourier[1] = pm
    fourier[N] = fourier[N-1] // std.revolve

    # Classical embeddings
    f.sign = __EMBED_SIGN__(f)
    f.xor = __EMBED_XOR__(f)
    f.inplace = __EMBED_INPLACE__(f)

    # Built-in functions
    id = '?' >> '?'
    discard = __DISCARD__()
    flip = std.flip
    measure = std.measure
