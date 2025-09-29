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

The rest of this README is dedicated to installation, basic testing, and
troubleshooting.

## Getting Started

### Dependencies

If you are working only on the AST, you need to install only [Rust][3].

Otherwise (e.g., to build the full compiler/runtime), you need to install
[Rust][3] _and_ the following:

1. LLVM 21.1.1 (with MLIR). First, download the LLVM build archive that is
   appropriate for your OS [from our repository][4]. Then you need to set both
   of the following environment variables (assuming `$HOME/bin` is where you
   extracted the LLVM archive, for example):
   ```
   $ export PATH=$PATH:$HOME/bin/llvm21/bin/
   $ export MLIR_DIR=$HOME/bin/llvm21/lib/cmake/mlir/
   ```
   You should set these persistently, e.g. in your `~/.bashrc` or `~/.zshrc`.

2. The following Debian/Ubuntu packages are also needed to build:
   ```
   $ sudo apt install build-essential cmake ninja-build zlib1g-dev libclang-dev
   ```
   If you are not a Debian/Ubuntu user, you can interpret `build-essential`
   above as "a C++ compiler."

The remaining steps on getting started depend on what portion of the compiler
you want to work on.

### Building Just the AST

If you just want to work on the AST, you can work on it in isolation as
follows:

    $ git submodule update --init qir_runner
    $ cd qwerty_ast
    $ cargo build

#### Coverage

To get coverage for the AST code, run:

    $ cargo llvm-cov --html

Then you can open `qwerty_ast/target/llvm-cov/html/index.html` (relative to the
repo root) in your browser.

You may need to install `cargo-llvm-cov` first with:

    $ cargo +stable install cargo-llvm-cov --locked

### Fiddling with MLIR

If you only want to work on MLIR (or the other C++ code in `qwerty_util`, or
even `tweedledum`):

    $ git submodule update --init tweedledum
    $ mkdir build && cd build
    $ cmake -G Ninja ..
    $ ninja

Note that the build you just created will not be used by Rust at all. It is
strictly for your separate personal enjoyment.

You can run the [FileCheck][5] tests by running the following command from the
root of the repository:

    $ python3 qwerty_mlir/tests/filecheck_tests.py

### Building the Python Extension

To generate the Python extension, run the following:

    $ git submodule update --init
    $ python3 -m venv venv
    $ . venv/bin/activate
    $ cd qwerty_pyrt
    $ pip install maturin numpy
    $ maturin develop

This will (re)build _everything_. Passing `-vv` to `maturin develop` can give
you an idea of what is going on.

To run the Python unit tests, say:

    $ python -m unittest qwerty.tests -v

To run _all_ tests across the compiler, run the following at the root of the
repository:

    $ dev/run-tests.sh

#### Tricks

To run a particular integration test in `gdb` (e.g., `bv_nometa`):

    $ gdb --args python3 -c 'import qwerty.tests.integ.nometa.bv; qwerty.tests.integ.nometa.bv.test(1)'

[3]: https://www.rust-lang.org/tools/install
[4]: https://github.com/gt-tinker/qwerty-llvm-builds/releases/tag/v21.1.1
[5]: https://llvm.org/docs/CommandGuide/FileCheck.html
