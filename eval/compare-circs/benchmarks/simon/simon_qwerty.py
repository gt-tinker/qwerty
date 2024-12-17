from qwerty import *
import qiskit.qasm3

def simon(f):
    @qpu[[N]](f)
    def kernel(f: cfunc[N]) -> bit[N]:
        return 'p'[N] + '0'[N] \
               | f.xor \
               | (std[N] >> pm[N]) + id[N] \
               | std[N].measure + discard[N]

    qasm = kernel.qasm()
    return qiskit.qasm3.loads(qasm), qasm

@classical[[K]]
def black_box(x: bit[2*K]) -> bit[2*K]:
    return x[:K], bit[1](0b0), x[K].repeat(K-1) ^ x[K+1:]

def get_circuit(n_qubits):
    # /2 due to K above being half the classical input size
    n_bits = n_qubits//2
    return simon(black_box[[n_bits]])
