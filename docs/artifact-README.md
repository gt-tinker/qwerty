CGO 2025 Round 2    
Artifact Evaluation    
Submission #68: ASDF: A Compiler for Qwerty, a Basis-Oriented Quantum Programming Language

------------------------------------------------------------------------------------------

This artifact is a compiler for the Qwerty quantum programming language.

Files in This Artifact
======================

1. `README.md`: This README with high-level instructions for using the artifact
2. `qwerty-artifact-docker.tar.xz`: Primary Docker image containing the Qwerty
   compiler/runtime (**Useful for artifact evaluation**)
3. `qwerty-artifact-quipper-docker.tar.xz`: Docker image for compiling Quipper
   to quantum assembly (**Useful for artifact evaluation**)
4. `qwerty-artifact-source.tar.xz`: Full source code for the artifact if one
   wants to inspect the source code, Docker is undesirable, or one wants to
   rebuild the Docker images from scratch.
   * This tarball contains additional documentation under `README.md` and
     `docs/*.md`. For example, `docs/project-structure.md` explains which
     sections of the paper map to which parts of the source code.
5. `llvm_mlir_rel_v19_1_2_x86_linux.tar.xz`: A build of LLVM 19.1.2 with MLIR.
   Useful for building the Qwerty compiler/runtime outside of the Docker image
   (but not strictly necessary as our work does not modify upstream LLVM)

Initial Setup
=============

You will need a Linux machine with at least 32 GB of RAM.

Download `qwerty-artifact-quipper-docker.tar.xz` and
`qwerty-artifact-docker.tar.xz`. Then import both docker images:

    $ docker image load -i qwerty-artifact-quipper-docker.tar.xz
    $ docker image load -i qwerty-artifact-docker.tar.xz

Run Evaluation
==============

Create a directory on the host to collect results:

    $ mkdir data

First, generate quantum assembly for Quipper benchmarks:

    $ docker run --rm -v $(pwd)/data/:/data/ qwerty-artifact-quipper quipper-bench-qasm.sh

Then run evaluation:

    $ docker run --rm -v $(pwd)/data:/data/ qwerty-artifact eval/run-eval.sh

This will take between 1.5 and 3 hours. On the host, check `data/summary/` for data
to compare with the paper. `table.csv` is Table 1; `time_*.pdf` are Fig. 11;
and `physical_*.pdf` are Fig. 12.

Run Tests
=========

The following command will run the test suite in the container:

    $ docker run --rm qwerty-artifact test/run-tests.sh
    [...voluminous output...]
    100% tests passed, 0 tests failed out of 1

    [...voluminous output...]
    Ran 41 tests in 10.390s

    OK

For more information on the test suite for the Qwerty compiler, see
`docs/testing.md` in the source tarball (or `/qwerty/docs/testing.md` in the
Docker image).

Run Example Programs
====================

There are many Qwerty examples in `examples/` in the source tarball (or
`/qwerty/examples` in the Docker image). The examples are described in
`docs/examples.md` in the source tarball (or `/qwerty/docs/examples.md` in the
Docker image).

To run an example program (e.g., Deutsch–Jozsa, implemented in `dj.py`), try:

    $ docker run --rm -w /qwerty/examples qwerty-artifact python dj.py
    Constant test:
    Classical: constant
    Quantum: constant

    Balanced test:
    Classical: balanced
    Quantum: balanced

Interactive Session
===================

The following command will start an interactive session inside the Docker image
(the subsequent commands show example commands to run in an interactive
session):

    $ docker run -it --rm qwerty-artifact bash
    # cd /qwerty/examples
    # python bv.py 1101
    Classical: 1101
    Quantum:   1101
    # cd /qwerty/test/
    # ./run-tests.sh
    [...voluminous output...]
    100% tests passed, 0 tests failed out of 1

    [...voluminous output...]
    Ran 41 tests in 10.390s

    OK

Building by Hand Outside Docker
===============================

Extract the tarball `qwerty-artifact-source.tar.xz`. The top-level `README.md`
contains build instructions. See `docs/eval.md` for details on running the
evaluation without the `qwerty-artifact` Docker image.

Software Licenses
=================

The Qwerty compiler is released under the MIT license (see `LICENSE` in the
source tarball), but other software is included in the source tarball and
Docker images, including:

1. [googletest][1], licensed under the BSD 3-clause license; see
   `tpls/googletest/LICENSE` in the source tarball
2. [qir-runner][2], licensed under the MIT license; see
   `tpls/qir-runner/LICENSE` in the source tarball
3. [qsharp][3], licensed under the MIT license; see
   `tpls/qsharp/LICENSE.txt` in the source tarball
4. [quipper-qasm][4], licensed under the GPLv3; see `tpls/quipper-qasm/LICENSE`
   in the source tarball. (I have received authorization from the author to
   distribute the code under the MIT license instead.)
5. [tweedledum][5], licensed under the MIT license; see `tpls/tweedledum/LICENSE`
   in the source tarball
6. LLVM, licensed under the [Apache 2.0 license (with LLVM exceptions)][6];
   see `/llvm/llvm19/include/llvm/Support/LICENSE.TXT` in the `qwerty-artifact`
   Docker image
7. .NET Core, licensed under the MIT license; see `/dotnet/LICENSE.txt` and
   `/dotnet/ThirdPartyNotices.txt` inside the `qwerty-artifact` Docker image
8. Rust, which is dual-licensed under MIT and Apache 2.
9. Various transitive Rust dependencies installed with `cargo`
10. Various Python dependencies installed with `pip`
11. Operating system packages whose licenses are available at
    `/usr/share/doc/*/copyright` inside both Docker images

[1]: https://github.com/google/googletest
[2]: https://github.com/qir-alliance/qir-runner/
[3]: https://github.com/microsoft/qsharp
[4]: https://github.com/ausbin/quipper-qasm/
[5]: https://github.com/boschmitt/tweedledum
[6]: https://llvm.org/LICENSE.txt
