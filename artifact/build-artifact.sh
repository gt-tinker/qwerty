#!/bin/bash
# This script builds the artifact to be uploaded to Zenodo. It will take about
# 20-30 minutes to run.

set -e

# Useful for security-minded individuals who heed the warning at the top of
# this page: https://wiki.debian.org/Docker. The default is for the more
# typical (and more insecure) configuration.
if [[ $DOCKER_USE_SUDO ]]; then
    docker_sudo=( sudo -g docker )
else
    docker_sudo=()
fi

whereami=$(readlink -f $(dirname "${BASH_SOURCE[0]}"))
upload_dir=$whereami/upload
repo_root=$whereami/..

mkdir -p "$upload_dir"

# Go to parent directory since that should be root of Docker build context (and
# it will make git happy to be in a git repo)
pushd "$repo_root"

# Check if we are in a git repo: https://stackoverflow.com/a/16925062/321301
if git rev-parse --is-inside-work-tree &>/dev/null; then
    printf '===> Building source tarball...\n' >&2
    git submodule update --init tpls/tweedledum tpls/qir-runner tpls/googletest tpls/qsharp tpls/quipper-qasm
    dest="$upload_dir/qwerty-artifact-source.tar"
    # Trick to get `git archive' to recurse into submodules:
    # https://stackoverflow.com/a/65466427/321301
    git archive --prefix=qwerty-artifact/ --output "$dest" @
    git submodule --quiet foreach --recursive 'git archive --prefix=qwerty-artifact/$displaypath/ -o submodule.tar @; tar Af '"$dest"' submodule.tar; rm submodule.tar'
    xz -T0 -f "$dest"
else
    printf '===> Skipping building source tarball since we are not in a git repo\n' >&2
fi

printf '===> Building Quipper image...\n' >&2
"${docker_sudo[@]}" docker build -f artifact/quipper.Dockerfile -t qwerty-artifact-quipper .
printf '===> Building Qwerty image...\n' >&2
"${docker_sudo[@]}" docker build -f artifact/qwerty.Dockerfile -t qwerty-artifact .

printf '===> Exporting and compressing Quipper image...\n' >&2
"${docker_sudo[@]}" docker save qwerty-artifact-quipper | xz -T0 >"$upload_dir/qwerty-artifact-quipper-docker.tar.xz"
printf '===> Exporting and compressing Qwerty image...\n' >&2
"${docker_sudo[@]}" docker save qwerty-artifact | xz -T0 >"$upload_dir/qwerty-artifact-docker.tar.xz"

printf '===> Copying README...\n' >&2
cp -v docs/artifact-README.md "$upload_dir/README.md"

printf '===> Downloading LLVM...\n' >&2

llvm_tarball=llvm_mlir_rel_v19_1_2_x86_linux.tar.xz
if [[ -e "$upload_dir/$llvm_tarball" ]]; then
    printf '===> Skipping downloading LLVM tarball since it already exists\n' >&2
else
    curl -fL "https://junk.ausb.in/qwerty/$llvm_tarball" -o "$upload_dir/$llvm_tarball"
fi
