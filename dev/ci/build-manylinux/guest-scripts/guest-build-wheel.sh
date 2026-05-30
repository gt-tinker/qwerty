#!/bin/bash
set -e -o pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s <plat> <repo-path> <wheel-out-dir>\n' "$0" >&2
    exit 1
fi
plat=$1
repo_path=$2
wheel_outdir=$3

. "$(dirname "${BASH_SOURCE[0]}")/guest-common.sh"

extract_llvm_if_needed

pushd "$wheel_outdir" >/dev/null
    rm -rvf *.whl
popd >/dev/null

pushd "$repo_path/qwerty_pyrt" >/dev/null
    maturin build --release --no-default-features --target-dir "$wheel_outdir" -o "$wheel_outdir" -vvv
popd >/dev/null

pushd "$wheel_outdir" >/dev/null
    linux_wheel=$(getwheel "$wheel_outdir" qwerty-*-manylinux_*.whl)

    auditwheel -vvv repair --plat "$plat" -w "$wheel_outdir" "$linux_wheel" |& tee auditwheel-log.txt

    manylinux_wheel=$(getwheel "$wheel_outdir" qwerty-*-manylinux_*.whl)
    printf '\nCreated manylinux wheel:\n%s\n' "$manylinux_wheel"
popd >/dev/null
