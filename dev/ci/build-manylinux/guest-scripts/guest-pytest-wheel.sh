#!/bin/bash
set -e -o pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <repo-path> <wheel-out-dir>\n' "$0" >&2
    exit 1
fi
repo_path=$1
wheel_outdir=$2

. "$(dirname "${BASH_SOURCE[0]}")/guest-common.sh"

pushd "$wheel_outdir" >/dev/null
    wheel=$(getwheel "$wheel_outdir" qwerty-*-manylinux_*.whl)

    rm -rf venv
    /opt/python/cp310-cp310/bin/python3 -m venv venv
    . venv/bin/activate
    pip install "$wheel"
popd >/dev/null

pushd "$repo_path/test" >/dev/null
    SKIP_FILECHECK_TESTS=1 python ./tests.py -v
popd >/dev/null
