These are some scripts for building [manylinux][1] (i.e., pretty portable)
builds of LLVM and the `qwerty` Python module (which includes the Qwerty
compiler). These make use of [manylinux Docker images][2], lightly modified to
include ninja and cargo.

You should run them in the following order:

1. To build a manylinux-friendly LLVM build:
   ```
   $ ./build-manylinux-llvm.sh 19.1.6
   ```
   Then the result will be in `io/llvm_mlir_rel_v19_1_6_x86_linux.tar.xz`.

2. To initially run the Qwerty compiler test suite:
   ```
   $ ./fulltest-manylinux-qwerty.sh
   ```
   This is not useful for packaging itself; instead, it is a good thing for CI
   to run to verify that you are not packaging up completely busted code.
   The compilation result of this will be thrown away since we don't want to
   include code for testing in the final wheel.

3. To compile a Qwerty manylinux wheel:
   ```
   $ ./build-manylinux-qwerty-wheel.sh
   ```
   This will produce a wheel at `io/qwerty-*-cp310-abi3-manylinux_*.whl`.

4. To run the integration tests on the built Qwerty manylinux wheel:
   ```
   $ ./pytest-manylinux-qwerty-wheel.sh
   ```
   This runs only the tests of the Python runtime, including the integration
   tests. (The other tests, such as the C++ unit tests or FileCheck tests,
   require test code that is intentionally excluded from the wheel.)

[1]: https://peps.python.org/pep-0600/
[2]: https://github.com/pypa/manylinux/
