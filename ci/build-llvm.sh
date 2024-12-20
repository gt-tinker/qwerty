#!/bin/bash
set -e

if [[ $# -lt 1 || $# -gt 3 ]]; then
    printf 'usage: %s <version> [<out-dir> [<llvm-dir>]]\n' "$0" >&2
    exit 1
fi

version=$1

if [[ $# -lt 2 ]]; then
    # Default to whatever directory this script is in
    relative_whereami=$(dirname "${BASH_SOURCE[0]}")
else
    relative_whereami=$2
fi

# `readlink -f' does not exist on the readlink that ships with macOS.
# Workaround: https://stackoverflow.com/a/70604668/321301
pushd "$relative_whereami" >/dev/null
    whereami=$(pwd -P)
popd >/dev/null

if [[ $# -lt 3 ]]; then
    # Default to out_dir/llvm
    repo_dir=$whereami/llvm
else
    # Same trick to mitigate missing `readlink -f' as above except if the
    # directory does not exist yet
    pushd "$(dirname "$3")" >/dev/null
        repo_dir=$(pwd -P)/$(basename "$3")
    popd >/dev/null
fi

kernel_name=$(uname -s)
case "$kernel_name" in
    Linux)
        os_name=linux ;;
    Darwin)
        os_name=macos ;;
    *)
        printf 'Unknown OS. Please fix the script\n' >&2
        exit 1
        ;;
esac

machine_name=$(uname -m)
case "$machine_name" in
    x86_64)
        arch_name=x86 ;;
    arm64)
        arch_name=aarch64 ;;
    *)
        printf 'Unknown architecture. Please fix the script\n' >&2
        exit 1
        ;;
esac

major_version=${version%%.*}
install_dir=$whereami/llvm$major_version
archive_filename=llvm_mlir_rel_v${version//./_}_${arch_name}_${os_name}.tar.xz

rm -rf "$install_dir"

if [[ ! -e $repo_dir ]]; then
    git clone https://github.com/llvm/llvm-project.git "$repo_dir"
fi

pushd "$repo_dir" >/dev/null
    git fetch origin
    git checkout "llvmorg-$version"
    rm -rf build
    mkdir build
    pushd build >/dev/null
        cmake -G Ninja -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=TRUE -DCMAKE_INSTALL_PREFIX="$install_dir" -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_INSTALL_UTILS=TRUE ../llvm
        time ninja install
    popd >/dev/null
popd >/dev/null

pushd "$install_dir/.." >/dev/null
    time XZ_OPT='-T0' tar -cJvf "$archive_filename" "$(basename "$install_dir")"
popd >/dev/null

printf 'Mission accomplished. Compressed archive path:\n'
printf '%s\n' "$whereami/$archive_filename"
