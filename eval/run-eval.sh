#!/bin/bash
# This file is a handy shortcut for running the steps described in the READMEs
# in the compare-circs/ and count-callables/ directories.

set -e

whereami="$(dirname "${BASH_SOURCE[0]}")"
summary_results_dir=${SUMMARY_RESULTS_DIR:-$whereami/results}
qre_results_dir=${QRE_RESULTS_DIR:-$whereami/compare-circs/results}

mkdir -p "$summary_results_dir" "$qre_results_dir"

printf '===> Running Resource Estimation\n'
pushd "$whereami/compare-circs/"
    ./qre.py
    ./merge-results.sh
    ./graph.py
popd
