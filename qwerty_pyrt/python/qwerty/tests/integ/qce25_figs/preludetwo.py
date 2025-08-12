# Basis macros
{bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
b.measure = __MEASURE__(b)
{bv1, bv2}.revolve = __REVOLVE__(bv1, bv2)

# More complicated bases
fourier[1] = pm
fourier[N] = fourier[N-1] // std.revolve

# Built-in functions
id = {'0','1'} >> {'0','1'}
discard = __DISCARD__()
flip = std.flip
measure = std.measure
