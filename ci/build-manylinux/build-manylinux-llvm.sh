#!/bin/bash
set -e -o pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <llvm-version>\n' "$0" >&2
    exit 1
fi
llvm_version=$1

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

run_docker "$guest_whereami_path_absolute/../build-llvm.sh" "$llvm_version" "$guest_io_path_absolute"
