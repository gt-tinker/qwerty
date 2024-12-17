#!/bin/bash
set -e
whereami=$(dirname "${BASH_SOURCE[0]}")
cd "$whereami"

# Skip the integration tests since they skew coverage
SKIP_INTEGRATION_TESTS=1 ./run-tests.sh --coverage "$@"

mkdir -p coverage-html/{cpp,py}/
# Python coverage report
coverage html
# C++ coverage report
gcovr

py_report=$(readlink -f coverage-html/py/index.html)
cpp_report=$(readlink -f coverage-html/cpp/index.html)

printf -- '-----------------------------\n\n'
printf 'Python coverage report: file://%s\n' "$py_report"
printf 'C++ coverage report: file://%s\n' "$cpp_report"
