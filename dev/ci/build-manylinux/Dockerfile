ARG PLAT=manylinux_2_28_x86_64
FROM quay.io/pypa/$PLAT

ARG NINJA_VERSION=1.12.1

ADD https://github.com/ninja-build/ninja/archive/refs/tags/v$NINJA_VERSION.tar.gz /tmp/ninja-$NINJA_VERSION.tar.gz
RUN tar -C /tmp -xvf /tmp/ninja-$NINJA_VERSION.tar.gz \
    && cd /tmp/ninja-$NINJA_VERSION \
    && mkdir build \
    && cd build \
    && cmake -G "Unix Makefiles" -DBUILD_TESTING=OFF .. \
    && make -j $(nproc) install \
    && rm -r /tmp/ninja-$NINJA_VERSION

RUN curl https://sh.rustup.rs -sSf | sh -s -- \
    -y \
    --profile minimal \
    --default-toolchain "1.80.1-x86_64-unknown-linux-gnu"

ENV PATH="${PATH}:/root/.cargo/bin/"

WORKDIR /io
