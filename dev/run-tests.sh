#!/bin/bash

# This script runs all the tests in the compiler.

whereami=$(dirname "${BASH_SOURCE[0]}")
repo_root=$whereami/..

if [[ -z $VIRTUAL_ENV ]]; then
    printf 'Hold your horses, partner. You need to activate the virtual '`
          `'environment. Try:\n\n' >&2
    printf '    . venv/bin/activate\n' >&2
    exit 1
fi

if [[ ! -e $whereami/bin ]]; then
    printf 'The qwerty-opt and qwerty-translate binaries are missing. Please '`
          `'run the maturin build.\n' >&2
    exit 1
fi

if ! which FileCheck &>/dev/null; then
    printf 'I could not locate LLVM FileCheck. Please ensure that the `bin` '`
          `'directory of an LLVM build is in your $PATH.\n' >&2
    exit 1
fi

ok=1

printf '\n=========> RUNNING AST TESTS\n\n'
pushd "$repo_root/qwerty_ast/" >/dev/null
    cargo test
    ret=$?
    (( ok = ok && !ret ))
popd >/dev/null

printf '\n=========> RUNNING MLIR TESTS\n\n'
pushd "$repo_root/qwerty_mlir/tests/" >/dev/null
    python filecheck_tests.py -v
    ret=$?
    (( ok = ok && !ret ))
popd >/dev/null

printf '\n=========> RUNNING RUNTIME & INTEGRATION TESTS\n\n'
pushd "$repo_root/qwerty_pyrt/" >/dev/null
    python -m unittest qwerty.tests -v
    ret=$?
    (( ok = ok && !ret ))
popd >/dev/null

if (( ok == 1 )); then
    printf '\nsuccess!\n'
    exit 0
else
    printf '\nSomething failed. Please look above.\n'
    exit 1
fi
