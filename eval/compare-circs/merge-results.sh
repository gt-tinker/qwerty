#!/bin/bash
set -e
shopt -s globstar

results_dir=${QRE_RESULTS_DIR:-results}

# Merge all the per-algorithm CSVs into one big CSV

# Only print the CSV header once (and remove spaces that confuse CSV readers).
# Then sort by optimization level, then problem size, then algorithm, then
# language, leaving the header line intact.
awk '{gsub(/ /, "")} /^Language,/ { if (!seen) { print $0; } seen=1; next } { print $0 }' "$results_dir"/**/resource_estimation.csv \
    | { sed -u '1q'; sort -k5,5 -k4,4n -k2,2 -k1,1 -t,; } \
    >"$results_dir"/results.csv
