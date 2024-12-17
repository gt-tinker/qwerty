from qwerty import *

set_func_opt(False)

def simon(f):
  @qpu[[N]](f)
  def kernel(f: cfunc[N]) -> bit[N]:
    return 'p'[N] + '0'[N] \
           | f.xor \
           | (std[N] >> pm[N]) + id[N] \
           | std[N].measure + discard[N]

  #return kernel()

@classical[[K]]
def black_box(x: bit[2*K]) -> bit[2*K]:
  return x[:K], bit[1](0b0), (x[K].repeat(K-1) ^ x[K+1:])

n_bits = 128
f = black_box[[n_bits//2]]
res = simon(f)
#print(res)
