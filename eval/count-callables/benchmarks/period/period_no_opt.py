from qwerty import *

set_func_opt(False)

def period_finding(black_box):
  @qpu[[M,N]](black_box)
  def kernel(black_box: cfunc[M,N]) -> bit[M]:
    return 'p'[M] + '0'[N] \
           | black_box.xor \
           | fourier[M].measure + discard[N]

  #result1, result2 = kernel(shots=2)
  #l_over_r1 = result1.as_bin_frac()
  #l_over_r2 = result2.as_bin_frac()
  #r = math.lcm(l_over_r1.denominator,
  #             l_over_r2.denominator)
  #return r

def get_black_box(n_bits_in,
                  n_bits_out,
                  n_mask_bits):
    @classical[[M,N,K]]
    def f(x: bit[M]) -> bit[N]:
      return bit[N-K](0b0), x[M-K:]

    return f[[n_bits_in,
              n_bits_out,
              n_mask_bits]]

problem_size = 128
n_bits_in = problem_size
n_bits_out = n_bits_in-1
n_masked = problem_size//2
black_box = get_black_box(n_bits_in,
                          n_bits_out,
                          n_masked)
res = period_finding(black_box)
#print(res)

