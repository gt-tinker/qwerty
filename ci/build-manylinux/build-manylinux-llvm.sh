#!/bin/bash
set -e -o pipefail

if [[ $# -ne 2 || ( ( $1 != docker ) && ( $1 != apptainer ) ) ]]; then
    printf 'usage: %s <docker|apptainer> <llvm-version>\n' "$0" >&2
    exit 1
fi
container=$1
llvm_version=$2

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

if [[ $container == docker ]]; then
    run_docker "$guest_whereami_path_absolute/../build-llvm.sh" "$llvm_version" "$guest_io_path_absolute"
elif [[ $container == apptainer ]]; then
    apptainer exec --bind "$repo_path_absolute:$guest_repo_path_absolute" "$rg_container_path" "$guest_whereami_path_absolute/../build-llvm.sh" "$llvm_version" "$guest_io_path_absolute"
else
    printf 'unknown type of container' >&2
    exit 1
fi
