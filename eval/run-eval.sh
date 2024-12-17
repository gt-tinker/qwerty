#!/bin/bash
# This file is a handy shortcut for running the steps described in the READMEs
# in the compare-circs/ and count-callables/ directories.

set -e

whereami="$(dirname "${BASH_SOURCE[0]}")"
summary_results_dir=${SUMMARY_RESULTS_DIR:-$whereami/results}
callable_results_dir=${CALLABLE_RESULTS_DIR:-$whereami/count-callables/results}
qre_results_dir=${QRE_RESULTS_DIR:-$whereami/compare-circs/results}

mkdir -p "$summary_results_dir" "$callable_results_dir" "$qre_results_dir"

printf '===> Counting QIR Callables (Section 8.2)\n'
pushd "$whereami/count-callables/"
    ./callables.py
    ./table.sh
    cp -v "$callable_results_dir/table.csv" "$summary_results_dir/"
popd

printf '===> Running Resource Estimation (Section 8.3)\n'
pushd "$whereami/compare-circs/"
    ./qre.py
    ./merge-results.sh
    ./graph.py

    for bench in bv grover simon period; do
        for metric in time physical; do
            cp -v "$qre_results_dir/${bench}_O3_${metric}.pdf" "$summary_results_dir/${metric}_${bench}.pdf"
        done
    done
popd
