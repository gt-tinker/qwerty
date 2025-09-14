from qwerty import *

def test(n_qubits, shots):
    @qpu
    def kernel_single_mask():
        return ('0'**n_qubits
                | {'0'**n_qubits, '1'**n_qubits}
                  >> {'1'**n_qubits, '0'**n_qubits}
                | measure**n_qubits)

    n_pad = n_qubits - 3

    @qpu
    def kernel_multi_mask():
        return ('1'**n_qubits
                | {'110' * '1'**n_pad, '111' * '1'**n_pad,
                   '101' * '1'**n_pad, '000' * '1'**n_pad}
                  >> {'101' * '1'**n_pad, '000' * '1'**n_pad,
                      '110' * '1'**n_pad, '111' * '1'**n_pad}
                | measure**n_qubits)

    return (kernel_single_mask(shots=shots),
            kernel_multi_mask(shots=shots))
