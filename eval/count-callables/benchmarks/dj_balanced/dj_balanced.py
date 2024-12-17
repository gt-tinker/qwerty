from qwerty import *

def deutsch_jozsa(f):
    @qpu[[N]](f)
    def kernel(f: cfunc[N,1]) -> bit[N]:
        return 'p'[N] | f.sign \
                      | pm[N].measure

    #return kernel()

@classical[[N]]
def balanced(x: bit[N]) -> bit:
  # f(x) = 1 for half the inputs
  # and f(x) = 0 for the other half
  return x.xor_reduce()

problem_size = 128
f = balanced[[problem_size]]
res = deutsch_jozsa(f)
#print(res)
