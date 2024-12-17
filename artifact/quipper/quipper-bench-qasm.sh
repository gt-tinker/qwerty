#!/bin/bash
set -e

bench_path=/benchmarks
data_path=/data/quipper
# >>> from benchmarks.grover import grover_qwerty
# >>> [grover_qwerty.get_n_iter(min(2**i, 8), 1) for i in range(10)]
# [0, 1, 3, 12, 12, 12, 12, 12, 12, 12]
grover_iters=( 0 1 3 12 12 12 12 12 12 12 )

run() {
    local args=( "$@" )
    local outname="$(printf -- '-%s' "${args[@]}")"
    outname=${outname:1}
    qasm.sh "$data_path/$outname.qasm" "${args[@]}"
    printf 'wrote %s\n' "$data_path/$outname.qasm"
}

run_all() {
    local i=$1
    local n_qubits=$(( 1 << i ))
    run bv.hs synth "$(printf '10%.0s' $(seq 1 $((n_qubits/2))))"
    run dj.hs constant synth "$n_qubits"
    run dj.hs balanced synth "$n_qubits"
    run grover.hs synth "$n_qubits" "${grover_iters[$i]}"
    run period.hs synth "$n_qubits" "$((n_qubits-1))" "$((n_qubits/2))"
    run simon.hs synth "$n_qubits"
}

mkdir -p "$data_path"
cd "$bench_path"
for i in {2..7}; do
    run_all "$i"
done
