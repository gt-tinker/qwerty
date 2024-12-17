from qwerty import *

@qpu
def teleport(secret: qubit) -> qubit:
    alice, bob = 'p0' | '1' & std.flip

    m_pm, m_std = secret + alice | '1' & std.flip \
                                 | (pm + std).measure

    secret_teleported = \
        bob | (pm.flip if m_std else id) \
            | (std.flip if m_pm else id)

    return secret_teleported

@qpu(teleport)
def kernel(teleport: qfunc) -> bit:
    # Try changing this to a different state
    example = 'j'
    return teleport(example) | ij.measure

def test(n_shots):
    return kernel(shots=n_shots, histogram=True)
