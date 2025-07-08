# Qwerty MLIR dialects

To build this, make sure `llvm-config` is in your `$PATH`. Then run:

    $ git submodule update --init tpls/tweedledum
    $ mkdir build && cd build
    $ cmake -G Ninja ..
    $ ninja
