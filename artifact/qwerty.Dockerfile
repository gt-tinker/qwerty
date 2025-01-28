#syntax=docker/dockerfile:1.5

FROM ubuntu:22.04 as toolchain

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    file \
    gcc \
    git \
    libssl-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libffi-dev \
    libzstd-dev \
    libtinfo-dev \
    libxml2-dev \
    vim \
    datamash \
    fonts-linuxlibertine

ADD https://junk.ausb.in/qwerty/llvm_mlir_rel_v19_1_2_x86_linux.tar.xz llvm.tar.xz
RUN mkdir -p /llvm \
    && tar -C /llvm -xJvf llvm.tar.xz \
    && rm llvm.tar.xz

ADD https://download.visualstudio.microsoft.com/download/pr/0e83f50a-0619-45e6-8f16-dc4f41d1bb16/e0de908b2f070ef9e7e3b6ddea9d268c/dotnet-sdk-6.0.302-linux-x64.tar.gz dotnet.tar.gz
RUN mkdir -p /dotnet \
    && tar -C /dotnet -xzvf dotnet.tar.gz \
    && rm dotnet.tar.gz

ENV PATH="${PATH}:/dotnet:/llvm/llvm19/bin/"
ENV MLIR_DIR="/llvm/llvm19/lib/cmake/mlir/"
ENV QWERTY_BUILD_TESTS=true

RUN curl https://sh.rustup.rs -sSf | sh -s -- \
    -y \
    --profile minimal \
    --default-toolchain "1.80.1-x86_64-unknown-linux-gnu"

ENV PATH="${PATH}:/root/.cargo/bin/"

COPY . /qwerty
WORKDIR /qwerty
RUN python3 -m venv /qwerty/venv
RUN . venv/bin/activate \
    && pip install -r eval/requirements.txt \
    && pip install -v tpls/qsharp/pip \
    && QWERTY_BUILD_TESTS=true pip install -v .

RUN mkdir /data/
ENV QUIPPER_QASM_DIR=/data/quipper/ \
    QRE_RESULTS_DIR=/data/compare-circs \
    CALLABLE_RESULTS_DIR=/data/count-callables \
    SUMMARY_RESULTS_DIR=/data/summary
WORKDIR /qwerty/

ENTRYPOINT ["/qwerty/artifact/entrypoint.sh"]
