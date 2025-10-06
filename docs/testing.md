Testing
=======

There are four ways we test the Qwerty compiler:

 1. Unit tests for Python code: `qwerty_pyrt/python/qwerty/tests/test_*.py`
    * Invoked with (before the second command, make sure the virtual
      environment is activated and you have run `maturin develop`):
      ```
      $ cd qwerty_pyrt/
      $ python -m unittest qwerty.tests -v
      ```
 2. Integration tests (running Qwerty code):
    `qwerty_pyrt/python/qwerty/integration_tests.py`, invoked same as #1 above
 3. Unit tests for Rust code: `qwerty_ast/**/test_*.rs`
    * Invoked with:
      ```
      $ cd qwerty_ast/
      $ cargo test
      ```
 4. IR-based MLIR tests: `qwerty_mlir/tests/**/*.mlir`
    * Invoked with (make sure you have [`FileCheck`][1] in your path):
      ```
      $ python qwerty_mlir/tests/filecheck_tests.py -v
      ```
      This script imitates [`lit`][2], the LLVM test runner.
    * These are usually for testing MLIR passes, but some MLIR [analyses][3]
      are tested too. These analyses are invoked with `qwerty-opt` via a
      boilerplate pass like `TestFuncSpecAnalysisPass`, which prints their
      results to stdout so that you and [`FileCheck`][1] can see them.

You can run all the tests with `dev/run-tests.sh`. Make sure you that have LLVM
binaries in your path (for [`FileCheck`][1]) and that you activated the virtual
environment (typically `. venv/bin/activate`).

[1]: https://llvm.org/docs/CommandGuide/FileCheck.html
[2]: https://llvm.org/docs/CommandGuide/lit.html
[3]: https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/
