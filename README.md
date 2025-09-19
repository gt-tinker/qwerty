# The (New) Qwerty Compiler

## Project Structure

This repository consists of the following four top-level projects:

1. `qwerty_ast` (Rust): Defines the Qwerty AST and typechecking/optimizations on it
2. `qwerty_mlir` (C++/Tablegen): MLIR dialects/passes for optimizing Qwerty
   programs and producing OpenQASM 3 or QIR
3. `qwerty_ast_to_mlir` (Rust): Converts a Qwerty AST to MLIR and JITs and runs it
4. `qwerty_pyrt` (Python/Rust): Defines the `qwerty` module, a little bit of Python that
   interfaces with the Rust code above via [PyO3][6]

There are also the following forks of third-party libraries that referenced as
git submodules:

1. [`qir_runner`][8] (Rust): Used for its implementation of the QIR runtime, which
   includes a good quantum simulator
2. `qwerty_mlir_sys` (Rust): A fork of [`mlir_sys`][1] that provides Rust
   bindings for the C API for MLIR dialects (both for our dialects and for
   upstream dialects)
3. `qwerty_melior` (Rust): A fork of [`melior`][2] a convenient wrapper for
   using MLIR APIs in Rust
4. `tblgen_rs` (Rust): A fork of [`tblgen_rs`][9], Rust bindings for
   [Tablegen][10] required by `melior` with no changes except upgrading LLVM
5. `eigen` (C++): Actually not a fork, just the [Eigen3][11] linear algebra
   library used by `qwerty_mlir`

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

If you only want to work on MLIR:

    $ git submodule update --init eigen
    $ mkdir build && cd build
    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
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
    $ pip install maturin
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

[1]: https://github.com/mlir-rs/mlir-sys/
[2]: https://github.com/mlir-rs/melior/
[3]: https://www.rust-lang.org/tools/install
[4]: https://github.com/gt-tinker/qwerty-llvm-builds/releases/tag/v21.1.1
[5]: https://llvm.org/docs/CommandGuide/FileCheck.html
[6]: https://pyo3.rs/
[8]: https://github.com/qir-alliance/qir-runner/
[9]: https://github.com/mlir-rs/tblgen-rs/
[10]: https://llvm.org/docs/TableGen/
[11]: https://eigen.tuxfamily.org/
