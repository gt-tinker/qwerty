Testing (need to fix the paths mentioned in this document)
=======

Types of tests and their locations:

1. Unit tests for Python code: `RuntimeTests` in `tests.py`
2. Unit tests for C++ code: `qwerty/**/*.cpp`, invoked by
   `ctest`/`qwerty-test`
3. Integration tests (running Qwerty code): `integration_tests/*.py` via
   `IntegrationTests` in `tests.py`
4. IR-based MLIR tests: `qwerty_mlir/**/*.mlir`, invoked by
   `FileCheckTests` in `tests.py`.
   * These are usually for testing MLIR passes, but some MLIR [analyses][4]
     are tested too. These analyses are invoked with `qwerty-opt` via a
     boilerplate pass like `TestFuncSpecAnalysisPass`, which prints their
     results to stdout so that you and [FileCheck][1] can see them.

You can run all the tests with `test/run-tests.sh` (or `test\run-tests.bat` on
Windows). Make sure you have LLVM binaries in your path (for [FileCheck][1]),
you activated the virtual environment (typically `. venv/bin/activate`), and
you built the tests by setting `$QWERTY_BUILD_TESTS` to true, i.e.,
`QWERTY_BUILD_TESTS=true pip install -v .`.

C++ Unit Tests
--------------
These are implemented with [GoogleTest][3].
You can run these on their own with

    $ ctest --test-dir _skbuild/ --output-on-failure --no-tests=error

Handy tips:
1. You can also run `qwerty-test` directly in the virtual environment to bypass
   `ctest`
2. ...or if you're very impatient (as I am), you can say
   `_skbuild/test/qwerty/qwerty-test` right after `pip install` finishes
   linking it, before `pip` slowly builds the wheel and installs the wheel.
3. If you want to run a test in gdb (say, the test `dimMismatch` in the test
   suite `qpuTypeCheckingBlit`), you can say
   ```
   $ gdb --args qwerty-test --gtest_filter=qpuTypeCheckingBlit.dimMismatch
   ```

Python Tests
------------
You can run these on their own by running `tests/test.py`.
For the sake of cross-platform compatibility, the Python tester also runs the
tests for MLIR passes with [FileCheck][1], which we enable by imitating
[lit][2], the LLVM test runner.

Coverage
--------
Update `pyproject.toml` to set `install.strip = false` and
`cmake.build-type = "Coverage"`. Then run

    $ rm -rv _skbuild
    $ QWERTY_BUILD_TESTS=true pip install -v .[coverage]
    $ test/coverage-report.sh

The last command will run all the tests and generate
`test/coverage-html/{cpp,py}/index.html` with coverage results. Note that it
skips the integration tests to avoid misleadingly inflating coverage.

[1]: https://llvm.org/docs/CommandGuide/FileCheck.html
[2]: https://llvm.org/docs/CommandGuide/lit.html
[3]: https://github.com/google/googletest
[4]: https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/
