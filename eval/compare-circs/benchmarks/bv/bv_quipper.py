import sys, os
import qiskit.qasm3
cur_dir = os.path.dirname(__file__)
hs_path = os.path.join(cur_dir, 'bv.hs')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from quipper_to_qasm import get_qasm_str

# Use the synthesized circuits since this is the most fair comparison with
# Qwerty, which implements the oracles with @classical (tweedledum synthesis)
IMPL = 'synth'

def get_circuit(secret_string):
    prog_args = [IMPL, secret_string]
    qasm_str = get_qasm_str(hs_path, prog_args, mute=True)
    return qiskit.qasm3.loads(qasm_str), qasm_str
