#!/bin/bash

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <llvm-version>\n' "$0" >&2
    exit 1
fi
llvm_version=$1

image_name=manylinux_2_28_x86_64-qwerty

# Useful for security-minded individuals who heed the warning at the top of
# this page: https://wiki.debian.org/Docker. The default is for the more
# typical (and more insecure) configuration.
if [[ $DOCKER_USE_SUDO ]]; then
    docker_sudo=( sudo -g docker )
else
    docker_sudo=()
fi

image_sha=$("${docker_sudo[@]}" docker images -q "$image_name")
if [[ -z $image_sha ]]; then
    "${docker_sudo[@]}" docker build -t "$image_name" .
fi

"${docker_sudo[@]}" docker run -it --rm -v $(pwd)/io:/io "$image_name" /usr/local/bin/llvm-build.sh "$llvm_version" /io
