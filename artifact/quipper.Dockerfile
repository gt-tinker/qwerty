# Taken from https://github.com/ausbin/docquipper/
# which in turn is based on https://github.com/spikecurtis/docquipper
# More details on what is happening in this file can be found in the README at
# the first link.

FROM haskell:8.6.5
ARG ARCHIVE_NAME=quipper-0.9.0.0

RUN cabal update
RUN cabal install quipper

RUN sed -i -e 's/deb.debian.org/archive.debian.org/g' \
           -e 's|security.debian.org|archive.debian.org/|g' \
           -e '/stretch-updates/d' /etc/apt/sources.list
RUN apt-get update && apt-get install -y gawk

ADD https://www.mathstat.dal.ca/~selinger/quipper/downloads/$ARCHIVE_NAME.tgz /tmp/
RUN tar -C /tmp -xvf /tmp/$ARCHIVE_NAME.tgz \
    && mkdir -p /root/.bin/ \
    && cp -a /tmp/$ARCHIVE_NAME/scripts/quipper /tmp/$ARCHIVE_NAME/scripts/convert_template.sh /tmp/$ARCHIVE_NAME/scripts/convert_template.awk /root/.bin/ \
    && rm -rvf /tmp/$ARCHIVE_NAME

COPY artifact/quipper/qasm.sh /root/.bin/qasm.sh
COPY artifact/quipper/quipper-bench-qasm.sh /root/.bin/quipper-bench-qasm.sh
COPY tpls/quipper-qasm /tmp/quipper-qasm/
WORKDIR /tmp/quipper-qasm/
RUN ghc QasmPrinting \
    && cp ./QasmPrinting /root/.bin/ \
    && rm -rvf /tmp/quipper-qasm

COPY eval/compare-circs/benchmarks/bv/bv.hs \
     eval/compare-circs/benchmarks/dj/dj.hs \
     eval/compare-circs/benchmarks/grover/grover.hs \
     eval/compare-circs/benchmarks/period/period.hs \
     eval/compare-circs/benchmarks/simon/simon.hs \
     /benchmarks/

ENV PATH=/root/.bin:$PATH

RUN mkdir /data/
WORKDIR /root/
