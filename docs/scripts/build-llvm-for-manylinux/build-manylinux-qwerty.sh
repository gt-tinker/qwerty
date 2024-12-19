#!/bin/bash
set -e -o pipefail

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

obliterate_build_files() {
    pushd "$repo_path_absolute"
        printf 'Using sudo to remove build files created as root in Docker...\n' >&2
        sudo rm -rf _skbuild tpls/qir-runner/target
    popd
}

obliterate_build_files
pushd "$io_path_absolute"
    sudo rm -rvf *.whl
popd

run_docker "$guest_whereami_path_absolute/guest-build-wheel.sh" "$plat" "$guest_repo_path_absolute" "$guest_io_path_absolute"

obliterate_build_files
