#!/bin/bash
set -e -o pipefail

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

pushd "$repo_path_absolute"
    sudo rm -rvf _skbuild tpls/qir-runner/target
popd

run_docker "$guest_whereami_path_absolute/guest-build-wheel.sh" "$plat" "$guest_repo_path_absolute" "$guest_io_path_absolute"
