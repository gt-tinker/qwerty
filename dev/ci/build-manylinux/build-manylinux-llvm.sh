#!/bin/bash
set -e -o pipefail

if [[ $# -lt 2 || ( ( $1 != docker ) && ( $1 != apptainer ) ) || $# -gt 3 ]]; then
    printf 'usage: %s <docker|apptainer> <llvm-version> [<llvm-repo>]\n' "$0" >&2
    exit 1
fi
container=$1
llvm_version=$2
llvm_repo_path=$3

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

args=( "$guest_whereami_path_absolute/../build-llvm.sh" "$llvm_version" "$guest_io_path_absolute" )

if [[ -n $llvm_repo_path ]]; then
    args+=( /io-llvm/ )

    pushd "$(dirname "$llvm_repo_path")" >/dev/null
        llvm_repo_path_absolute=$(pwd -P)/$(basename "$llvm_repo_path")
    popd >/dev/null
fi

if [[ $container == docker ]]; then
    if [[ -n $llvm_repo_path ]]; then
        printf 'Passing an LLVM repo path with Docker is not supported right now '`
              `'because I have not needed it yet\n' >&2
        exit 1
    fi

    run_docker "${args[@]}"
elif [[ $container == apptainer ]]; then
    bind_arg=$repo_path_absolute:$guest_repo_path_absolute

    if [[ -n $llvm_repo_path ]]; then
        bind_arg+=,$llvm_repo_path_absolute:/io-llvm/
    fi

    apptainer exec --bind "$bind_arg" "$rg_container_path" "${args[@]}"
else
    printf 'unknown type of container' >&2
    exit 1
fi
