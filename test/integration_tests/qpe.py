from qwerty import *

def qpe(precision, prepU, Upow2pow, n_shots):
    @qpu[[M,T]](prepU, Upow2pow)
    def kernel(prepU: func[[],qubit[M]],
               Upow2pow: rev_qfunc[M][[...]]) -> bit[T]:
        return 'p'[T] + prepU() \
               | (std[T-1-j] + '1' + std[j] & Upow2pow[[j]]
                  for j in range(T)) \
               | fourier[T].measure + discard[M]

    k_inst = kernel[[precision]]
    for meas in k_inst(shots=n_shots):
        yield meas.as_bin_frac()
