#!/bin/bash
set -e
whereami=$(dirname "${BASH_SOURCE[0]}")

if [[ $1 = '--coverage' ]]; then
    shift
    py_run_cmd=( coverage run )
else
    py_run_cmd=( python -Wignore::SyntaxWarning )
fi

if [[ -n $1 ]]; then
    py_filter=( -k "$1" )
else
    py_filter=()
fi

if [[ -n $SKBUILD_BUILD_DIR ]]; then
    skbuild_dir=$SKBUILD_BUILD_DIR
else
    skbuild_dir=$whereami/../_skbuild
fi

# C++ unit tests
ctest -V --test-dir "$skbuild_dir" --output-on-failure --no-tests=error

# Python integration tests and FileCheck (*.mlir) tests
"${py_run_cmd[@]}" "$whereami/tests.py" -v "${py_filter[@]}"
