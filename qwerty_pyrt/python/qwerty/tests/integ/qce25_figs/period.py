"""
Fig. 16 of the Qwerty QCE '25 paper.
"""

import math
from fractions import Fraction
from qwerty import *

def period_finding(f):
  @qpu[[N]]
  def kernel():
    return ('p'**N * '0'**N
            | f.xor
            | id**N * discard**N
            | fourier[N].measure)

  def shift_binary_point(bits):
    return Fraction(int(bits),
                    2**len(bits))

  frac1 = shift_binary_point(kernel())
  frac2 = shift_binary_point(kernel())
  return math.lcm(frac1.denominator,
                  frac2.denominator)

@classical
def mod4(x: bit[3]) -> bit[3]:
  return x % 4

#period = period_finding(mod4)
#
#if period == 4:
#  print('Success!')
#else:
#  print('Period finding failed')

def test():
  period = period_finding(mod4)
  
  if period == 4:
    return 'Success!'
  else:
    return 'Period finding failed'
