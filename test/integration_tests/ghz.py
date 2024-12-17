from qwerty import *

def test(n_qubits):
    @qpu[[N]]
    def kernel() -> bit[N]:
        return '+' + '0'[N-1] | '1' & std.flip[N-1] \
                              | std[N].measure

    return kernel[[n_qubits]](histogram=True, shots=2048)
