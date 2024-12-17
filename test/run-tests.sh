#!/bin/bash
set -e
whereami=$(dirname "${BASH_SOURCE[0]}")

# C++ unit tests
ctest -V --test-dir "$whereami/../_skbuild/" --output-on-failure --no-tests=error

# Python integration tests and FileCheck (*.mlir) tests
python -Wignore::SyntaxWarning "$whereami/tests.py" -v
