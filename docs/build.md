Building a Subset of the Qwerty Compiler
========================================

If you are working on a specific part of the Qwerty compiler (e.g., if you are
just modifying typechecking or just changing circuit synthesis), it can be
useful to build and test only the relevant component. This can often bypass
slow linking steps and produce cleaner, more direct output rather than being
filtered through multiple levels of `maturin` and `cmake`.

(If you want to build the full compiler, see [the README](../README.md)
instead.)

Building Only the AST
---------------------

Building only the AST is helpful if you want to modify e.g. typechecking or
metaQwerty expansion in isolation. You can test your work only with unit tests.

The only dependency needed is [Rust][1]. You can build the AST code in
isolation (i.e., without MLIR or the Python runtime) as follows:

    $ git submodule update --init qir_runner
    $ cd qwerty_ast
    $ cargo build

To run unit tests:

    $ cargo test

### Coverage

To get coverage for the AST code, run:

    $ cargo llvm-cov --html

Then you can open `qwerty_ast/target/llvm-cov/html/index.html` (relative to the
repo root) in your browser.

You may need to install `cargo-llvm-cov` first with:

    $ cargo +stable install cargo-llvm-cov --locked

Building Only MLIR
------------------

Building only the MLIR portion of the compiler is useful if, for example, you
are working on a compiler pass and want to test that pass on example IR rather
than by running an end-to-end test from Qwerty code.

You need neither Rust nor a Python virtual environment for this. However, you
do need LLVM/MLIR and the C++ libraries as mentioned in [the
README](../README.md). Once you have the dependencies installed, run the
following commands from the root of the repository:

    $ git submodule update --init tweedledum
    $ mkdir build && cd build
    $ cmake -G Ninja ..
    $ ninja

It is crucial to run the configuration step (the `cmake ..` command) in a
separate `build` directory as shown above, not directly in the root of the
repository. (Otherwise, CMake will get very confused when building the Python
extension as described in the [README](../README.md).)

Otherwise, note that build you just created will not be used by Rust at all. It
is separate. You can run an MLIR pass with `bin/qwerty-opt`. 

To run the [FileCheck][2] tests, run the following command:
repository:

    $ python3 ../qwerty_mlir/tests/filecheck_tests.py

[1]: https://www.rust-lang.org/tools/install
[2]: https://llvm.org/docs/CommandGuide/FileCheck.html
