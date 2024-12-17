from qwerty import *
import math

from .qpe import qpe

def order_finding(epsilon, x, modN):
    L = math.ceil(math.log2(modN))
    t = 2*L + 1 + math.ceil(math.log2(2+1/(2*epsilon)))

    @qpu[[M]]
    def prepU() -> qubit[M]:
        return '0'[M-1] + '1'

    @classical[[X,N,M,J]]
    def xymodN(y: bit[M]) -> bit[M]:
        return X**2**J * y % N

    x_inv = pow(x, -1, modN)
    multiplier = xymodN[[x,modN,L,...]].inplace(xymodN[[x_inv,modN,L,...]])
    binfrac1, binfrac2 = qpe(t, prepU, multiplier, n_shots=2)

    def denom(frac):
        cf = cfrac.from_fraction(frac)
        for c in reversed(cf.convergents()):
            if c.denominator < modN:
                return c.denominator

    r = math.lcm(denom(binfrac1),
                 denom(binfrac2))

    if x**r % modN == 1:
        return r
    else:
        raise ValueError('need to retry order finding')
