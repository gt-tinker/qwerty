"""
Fig. 18 of the Qwerty QCE '25 paper.
"""

import math
from qwerty import *
#from qpe import qpe
from .qpe2 import qpe

def order_finding(err_tol, x, modN):
  m = math.ceil(math.log2(modN))
  prec = 2*m + 1 + math.ceil(
    math.log2(2+1/(2*err_tol)))

  @qpu
  def one():
    return '0'**(m-1) * '1'

  @classical[[J]]
  @reversible
  def mult(y: bit[m]) -> bit[m]:
    return x**2**J * y % modN

  op = mult.inplace
  frac1 = qpe(prec, one, op)
  frac2 = qpe(prec, one, op)

  def get_denom(frac):
    cf = cfrac(frac)
    for conv in reversed(cf.convergents()):
      if conv.denominator < modN:
        return conv.denominator

  return math.lcm(get_denom(frac1),
                  get_denom(frac2))

#print(order_finding(0.2, 7, 15))

def test():
  return str(order_finding(0.2, 7, 15))
