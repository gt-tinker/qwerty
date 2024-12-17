from qwerty import *

set_func_opt(False)

def bv(f):
  @qpu[[N]](f)
  def kernel(f: cfunc[N,1]) -> bit[N]:
    return 'p'[N] | f.sign \
                  | pm[N] >> std[N] \
                  | std[N].measure

  #return kernel()

def get_black_box(secret_string):
  @classical[[N]](secret_string)
  def f(secret_string: bit[N],
        x: bit[N]) -> bit:
    return (secret_string & x) \
           .xor_reduce()

  return f

input = '10'*(128//2)
secret_str = \
  bit.from_str(input)
n_bits = len(secret_str)
black_box = get_black_box(secret_str)
res = bv(black_box)
#print(res)
