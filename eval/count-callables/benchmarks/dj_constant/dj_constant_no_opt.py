from qwerty import *

set_func_opt(False)

def deutsch_jozsa(f):
    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] | f.sign \
                      | pm[N].measure

    #return kernel()

@classical[[N]]
def constant(x: bit[N]) -> bit:
  # f(x) = 1
  return bit[1](0b1)

problem_size = 128
f = constant[[problem_size]]
res = deutsch_jozsa(f)
#print(res)
