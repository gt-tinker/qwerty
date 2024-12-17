from qwerty import *

# This is Fig. 5 from the CGO paper
@qpu
@reversible
def swap_by_renaming(q: qubit[4]) -> qubit[4]:
    q0, q1, q2, q3 = q
    return q0 + (q1 + q3 + q2 | std[3] >> {'1','0'}[3])

def test(init_state, n_shots):
    @qpu(init_state, swap_by_renaming)
    def kernel(init_state: bit[7], swap_by_renaming: rev_qfunc[4]) -> bit[7]:
        return init_state.q | {'101', '010'} & swap_by_renaming | std[7].measure

    return kernel(shots=n_shots, histogram=True)

