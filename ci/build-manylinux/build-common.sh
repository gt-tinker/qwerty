# Common code for build-manylinux-llvm.sh, build-manylinux-qwerty-wheel.sh, and
# other similar scripts in this directory

repo_path_absolute=$(git rev-parse --show-toplevel)
pushd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null
    whereami_relative=$(git rev-parse --show-prefix)
popd >/dev/null

whereami_absolute=$repo_path_absolute/$whereami_relative
dockerfile_path=$whereami_absolute/Dockerfile
plat=manylinux_2_28_x86_64
image_name=$plat-qwerty
rg_container_dir=/projects/ci-runners/qwerty-llvm-builds/
rg_container_path=/projects/ci-runners/qwerty-llvm-builds/$image_name.sif

# Useful for security-minded individuals who heed the warning at the top of
# this page: https://wiki.debian.org/Docker. The default is for the more
# typical (and more insecure) configuration.
if [[ $DOCKER_USE_SUDO ]]; then
    docker_sudo=( sudo -g docker )
else
    docker_sudo=()
fi

io_path_absolute=$whereami_absolute/io
mkdir -p "$io_path_absolute"
guest_repo_path_absolute=/io/
guest_whereami_path_absolute=$guest_repo_path_absolute/$whereami_relative
guest_io_path_absolute=$guest_whereami_path_absolute/io
guest_script_path_absolute=$guest_whereami_path_absolute/guest-scripts

build_docker_if_needed() {
    image_sha=$("${docker_sudo[@]}" docker images -q "$image_name")
    if [[ -z $image_sha ]]; then
        "${docker_sudo[@]}" docker build -t "$image_name" -f "$dockerfile_path" --build-arg PLAT="$plat" "$repo_path_absolute"
    fi
}

run_docker() {
    build_docker_if_needed

    "${docker_sudo[@]}" docker run --rm -v "$repo_path_absolute:$guest_repo_path_absolute" "$image_name" "$@"
}
