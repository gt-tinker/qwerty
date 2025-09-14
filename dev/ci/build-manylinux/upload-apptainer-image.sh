#!/bin/bash
set -e -o pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <rg-ssh-hostname>\n' "$0" >&2
    exit 1
fi
host=$1

. "$(dirname "${BASH_SOURCE[0]}")/build-common.sh"

build_docker_if_needed

image_path=$image_name.sif

rm -vf "$image_path"
"${docker_sudo[@]}" apptainer build "$image_path" "docker-daemon://$image_name:latest"

scp "$image_path" "$host:$rg_container_path"

# Free up some disk space
rm -vf "$image_path"
