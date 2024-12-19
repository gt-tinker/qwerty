FROM quay.io/pypa/manylinux_2_28_x86_64
ARG NINJA_VERSION=1.12.1

ADD https://github.com/ninja-build/ninja/archive/refs/tags/v$NINJA_VERSION.tar.gz /tmp/ninja-$NINJA_VERSION.tar.gz
RUN tar -C /tmp -xvf /tmp/ninja-$NINJA_VERSION.tar.gz \
    && cd /tmp/ninja-$NINJA_VERSION \
    && mkdir build \
    && cd build \
    && cmake -G "Unix Makefiles" -DBUILD_TESTING=OFF .. \
    && make -j $(nproc) install \
    && rm -r /tmp/ninja-$NINJA_VERSION

COPY llvm-build.sh /usr/local/bin/llvm-build.sh

WORKDIR /io
