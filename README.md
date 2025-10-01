The Compiler for Qwerty, a Basis-Oriented Quantum Programming Language
======================================================================

This is the compiler (and examples) for Qwerty, a quantum programming language
embedded in Python. It is licensed under the MIT license. If you want to contribute
or have issues, you can check out [`CONTRIBUTING.md`](CONTRIBUTING.md).

Documentation
-------------

The `docs/` directory contains more documentation that does not fit in this
README:

* [`docs/project-structure.md`](docs/project-structure.md): An overview
  of the contents of this project (i.e., which files do what) and how the
  sections in the paper submission map to source files.
* [`docs/examples.md`](docs/examples.md): A list of the example
  programs found in the `examples/` directory.
* [`docs/build.md`](docs/build.md): Instructions for building subsets of the
  Qwerty compiler when working on particular components.
* [`docs/upgrading-llvm.md`](docs/upgrading-llvm.md): Describes the
  semi-automated process for upgrading the version of LLVM used by the Qwerty
  compiler.
* [`docs/new-mlir-attr-rust.md`](docs/new-mlir-attr-rust.md): Demonstrates the
  interplay betwen Tablegen, C++, C, and Rust needed to add a new attribute to
  one of our MLIR dialects.

The rest of this README is dedicated to installation, basic testing, and
troubleshooting.

Getting Started
---------------

This guide describes the common case of building the full Qwerty
compiler/runtime. (For instructions on building some portions of the compiler
independently, see [`docs/build.md`](docs/build.md).)

### Dependencies

You need to install [Rust][1] _and_ the following:

1. LLVM 21.1.1 (with MLIR). First, download the LLVM build archive that is
   appropriate for your OS and architecture [from our repository][2]. Then you
   need to set both of the following environment variables (assuming
   `$HOME/bin` is where you extracted the LLVM archive, for example):
   ```
   $ export PATH=$PATH:$HOME/bin/llvm21/bin/
   $ export MLIR_DIR=$HOME/bin/llvm21/lib/cmake/mlir/
   ```
   You should set these persistently, e.g., in your `~/.bashrc` or `~/.zshrc`.

2. The following Debian/Ubuntu packages are also needed to build:
   ```
   $ sudo apt install build-essential cmake ninja-build zlib1g-dev libclang-dev
   ```
   If you are on macOS, installing the XCode developer tools is enough and the
   command above is not needed.

### Building

To generate the Python extension, run the following:

    $ git submodule update --init
    $ python3 -m venv venv
    $ . venv/bin/activate
    $ cd qwerty_pyrt
    $ pip install maturin numpy
    $ maturin develop -vvv

The last command will (re)build _everything_. Passing `-vvv` to
`maturin develop` helps to give you an idea of what is going on.

### Testing

As a quick smoke test, you can run a single Qwerty program from `examples/` as
follows:

    $ python ../examples/deutsch.py
    Balanced f:
    Classical: f(0) xor f(1) = 1
    Quantum:   f(0) xor f(1) = 1

    Constant f:
    Classical: f(0) xor f(1) = 0
    Quantum:   f(0) xor f(1) = 0

To run _all_ tests across the compiler, run the following:

    $ ../dev/run-tests.sh

Troubleshooting
---------------

If your compilation process keeps getting sniped by the OOM killer, as seen
below:

    c++: fatal error: Killed signal terminated program cc1plus
      compilation terminated.

Then a potential fix is to add the following code near the top of your
`CMakeLists.txt` both in `/` and  `/tweedledum`:

    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=1)
    set(CMAKE_JOB_POOL_LINK link_job_pool)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=1)
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)

This tells CMake to tell Ninja to limit the number of linking and compilation
jobs done in parallel to just 1 each, although this can be changed by changing
the above parameters.

Citation
--------

To cite the Qwerty compiler, can cite our CGO '25 paper:

Austin J. Adams, Sharjeel Khan, Arjun S. Bhamra, Ryan R. Abusaada, Anthony M.
Cabrera, Cameron C. Hoechst, Travis S. Humble, Jeffrey S. Young, and Thomas M.
Conte. March 2025. **"ASDF: A Compiler for Qwerty, a Basis-Oriented Quantum
Programming Language."** In _Proceedings of the 23rd ACM/IEEE International
Symposium on Code Generation and Optimization (CGO '25)_.
https://doi.org/10.1145/3696443.3708966

BibTeX citation:

    @inproceedings{adams_asdf_2025,
        author = {Adams, Austin J. and Khan, Sharjeel and Bhamra, Arjun S. and Abusaada, Ryan R. and Cabrera, Anthony M. and Hoechst, Cameron C. and Humble, Travis S. and Young, Jeffrey S. and Conte, Thomas M.},
        title = {ASDF: A Compiler for Qwerty, a Basis-Oriented Quantum Programming Language},
        year = {2025},
        isbn = {9798400712753},
        url = {https://doi.org/10.1145/3696443.3708966},
        doi = {10.1145/3696443.3708966},
        booktitle = {Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization},
        series = {CGO '25}
        pages = {444–458},
        numpages = {15},
    }


The evaluation for the CGO '25 paper uses additional code not on the `main`
branch of this repository. You can find that evaluation code in [the Zenodo
artifact][3] or on the `cgo25-artifact` branch.

[1]: https://www.rust-lang.org/tools/install
[2]: https://github.com/gt-tinker/qwerty-llvm-builds/releases/tag/v21.1.1
[3]: https://doi.org/10.5281/zenodo.14080494
