from qwerty import *

@qdefs
def builtin_defs():
    'p' <= '0' or '1'
    'm' <= '0' or -'1'
    'i' <= '0' or '1'@90
    'j' <= '0' or '1'@270

    std <= {'0', '1'}
    pm <= {'p', 'm'}
    ij <= {'i', 'j'}
    fourier[N] <= {['0' or '1'@(360*(j%(2**(k+1)))/2**(k+1)) for k in range(N)]
                   for j in range(2**N)}

@qpu[[N]]
def kern() -> bit[N]:
    return 'p'[N] | fourier[N].measure

print_histogram(kern(histogram=True, shots=1024))
