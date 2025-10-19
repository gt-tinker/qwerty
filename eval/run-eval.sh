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

    #for bench in bv grover simon period; do
    #    for metric in time physical; do
    #        cp -v "$qre_results_dir/${bench}_O3_${metric}.pdf" "$summary_results_dir/${metric}_${bench}.pdf"
    #    done
    #done
popd
