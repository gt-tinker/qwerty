# Common code for guest-{build-wheel,test-wheel}.sh
# Expects $wheel_outdir to be defined

getwheel() {
    # Just for error messages
    local wheel_dir=$1
    shift
    local wheels=( "$@" )
    if [[ ${#wheels[@]} -eq 0 ]]; then
        printf 'Could not find a wheel in the directory %s\n' "$wheel_dir" >&2
        exit 1
    elif [[ ${#wheels[@]} -gt 1 ]]; then
        printf 'Too many wheels in the directory %s\n' "$wheel_dir" >&2
        exit 1
    fi
    local wheel=${wheels[0]}
    printf '%s' "$wheel"
}

extract_llvm_if_needed() {
    pushd "$wheel_outdir" >/dev/null
        if [[ ! -e llvm20 ]]; then
            time tar -xvf llvm_mlir_rel_v20_1_6_x86_linux.tar.xz
        fi
    popd >/dev/null

    export PATH=$wheel_outdir/llvm20/bin:$PATH
    export MLIR_DIR=$wheel_outdir/llvm20/lib/cmake/mlir
}

export SKBUILD_BUILD_DIR=$wheel_outdir/_skbuild
