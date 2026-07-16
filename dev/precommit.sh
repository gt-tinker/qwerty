#!/bin/bash

# This script runs pre-commit checks on the repository.
# It formats all first-party Rust crates using cargo fmt.

whereami=$(dirname "${BASH_SOURCE[0]}")
repo_root=$whereami/..

ok=1

RUST_CRATES=(
    qwerty_ast_macros
    qwerty_ast
    qwerty_ast_to_mlir
    qwerty_mlir_sys
    qiree_sys
    qwerty_pyrt/rust
)

for crate in "${RUST_CRATES[@]}"; do
    printf '\n=========> RUNNING cargo fmt ON %s\n\n' "$crate"
    pushd "$repo_root/$crate" > /dev/null
        cargo fmt
        ret=$?
        (( ok = ok && !ret ))
    popd > /dev/null
done

if (( ok == 1 )); then
    printf '\nsuccess!\n'
    exit 0
else
    printf '\nSomething failed. Please look above.\n'
    exit 1
fi
