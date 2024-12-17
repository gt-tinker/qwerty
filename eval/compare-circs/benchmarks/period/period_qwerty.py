from qwerty import *
import qiskit.qasm3

def period_finding(black_box):
    @qpu[[M,N]](black_box)
    def kernel(black_box: cfunc[M,N]) -> bit[M]:
        return 'p'[M] + '0'[N] \
               | black_box.xor \
               | fourier[M].measure + discard[N]

    return kernel.qasm()

def get_black_box(n_bits_in,
                  n_bits_out,
                  n_mask_bits):
    @classical[[M,N,K]]
    def f(x: bit[M]) -> bit[N]:
        return bit[1](0b1).repeat(N-K), x[M-K:]

    return f[[n_bits_in,
              n_bits_out,
              n_mask_bits]]

def get_circuit(n_bits_in, n_bits_out, n_mask_bits):
    qasm = period_finding(get_black_box(n_bits_in, n_bits_out, n_mask_bits))
    return qiskit.qasm3.loads(qasm), qasm


