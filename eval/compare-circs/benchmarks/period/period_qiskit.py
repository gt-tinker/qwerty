import qiskit.qasm2
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def get_circuit(n_bits_in, n_bits_out, n_mask_bits):
    period_circuit = QuantumCircuit(n_bits_in + n_bits_out, n_bits_in)

    # initialize inputs in state |+>
    period_circuit.h(range(n_bits_in))

    x_start = 0
    x_end = x_start + n_bits_in
    y_start = x_end
    y_end = y_start + n_bits_out

    # Leading 11111111...1
    period_circuit.x(range(y_start, y_end - n_mask_bits))
    # Copy last n_mask_bits bits from x to y
    period_circuit.cx(range(x_end - n_mask_bits, x_end),
                      range(y_end - n_mask_bits, y_end))

    iqft = QFT(num_qubits=n_bits_in, inverse=True)
    period_circuit.append(iqft, range(x_start, x_end))
    period_circuit.measure(range(x_start, x_end), range(n_bits_in))

    return period_circuit, qiskit.qasm2.dumps(period_circuit)
