#!/bin/bash
set -e -o pipefail

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

run_docker "$guest_script_path_absolute/guest-fulltest.sh" "$guest_repo_path_absolute" "$guest_io_path_absolute"
