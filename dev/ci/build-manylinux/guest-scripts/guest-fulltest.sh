#!/bin/bash
set -e -o pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <repo-path> <wheel-out-dir>\n' "$0" >&2
    exit 1
fi
repo_path=$1
wheel_outdir=$2

. "$(dirname "${BASH_SOURCE[0]}")/guest-common.sh"

extract_llvm_if_needed

pushd "$wheel_outdir" >/dev/null
    rm -rf venv
    /opt/python/cp310-cp310/bin/python3 -m venv venv
    . venv/bin/activate
popd >/dev/null

pushd "$repo_path" >/dev/null
    QWERTY_BUILD_TESTS=1 pip install -v .
    pushd test >/dev/null
        ./run-tests.sh
    popd >/dev/null
popd >/dev/null
