#!/bin/bash
set -e -o pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s <plat> <repo-path> <wheel-out-dir>\n' "$0" >&2
    exit 1
fi
plat=$1
repo_path=$2
wheel_outdir=$3

pushd "$wheel_outdir"
    if [[ ! -e llvm19 ]]; then
        time tar -xvf llvm_mlir_rel_v19_1_6_x86_linux.tar.xz
    fi
popd

export PATH=$wheel_outdir/llvm19/bin:$PATH
export MLIR_DIR=$wheel_outdir/llvm19/lib/cmake/mlir

pushd "$repo_path"
    /opt/python/cp310-cp310/bin/pip wheel --no-deps -w "$wheel_outdir" -v .
popd

pushd "$wheel_outdir"
    wheels=( qwerty-*.whl )
    if [[ ${#wheels[@]} -ne 1 ]]; then
        printf 'Could not find a wheel named qwerty-*.whl in the directory %s\n' "$wheel_outdir" >&2
        exit 1
    fi
    wheel=${wheels[0]}

    auditwheel -vvv repair --plat "$plat" -w "$wheel_outdir" "$wheel" |& tee auditwheel-log.txt
popd
