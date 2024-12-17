@echo off

rem C++ unit tests
ctest -V --test-dir %~dp0..\_skbuild\ --output-on-failure --no-tests=error

rem Python integration tests and FileCheck (*.mlir) tests
python %~dp0tests.py -v
