#!/bin/bash
set -e
shopt -s globstar

results_dir=${CALLABLE_RESULTS_DIR:-results}

mkdir -p "$results_dir"

# Merge all the per-algorithm CSVs into one big CSV.
# Only print the CSV header once (and remove spaces that confuse CSV readers).
# Then sort appropriately, leaving the header line intact.
awk '{gsub(/ /, "")} /^Language,/ { if (!seen) { print $0; } seen=1; next } { print $0 }' benchmarks/**/callables_count.csv \
    | { sed -u '1q'; sort -k2,2 -k3,3 -k1,1 -t,; } \
    | tee "$results_dir/callables_count.csv" \
    | grep -v Constant \
    | cut -d , -f 1,2,4,6 \
    | sed -e 's/__quantum__rt__callable_//g' -e 's/Qsharp/Q#/g' -e 's/Bernstein-Vazirani/B--V/g' -e 's/Deutsch-Jozsa/D--J/g' -e 's/invoke/inv./g' -e 's/Qwerty-No-Opt/Asdf (No Opt)/g' -e 's/Qwerty-Opt/Asdf (Opt)/g' \
    | tee "$results_dir/callables_count_trim.csv" \
    | datamash -Ht, -s crosstab 2,1 sum 3 \
    > "$results_dir/pivot_create.csv"

# Construct the second pivot table
datamash -Ht, -s crosstab 2,1 sum 4 < "$results_dir/callables_count_trim.csv" \
    > "$results_dir/pivot_inv.csv"

# Interleave them now
{
    printf ',Q#,,Asdf (No Opt),,Asdf (Opt),\n'
    printf ',create,inv.,create,inv.,create,inv.\n'

    paste -d, "$results_dir/pivot_create.csv" "$results_dir/pivot_inv.csv" \
        | awk -F, 'NR > 2 {print $1 "," $4 "," $8 "," $2 "," $6 "," $3 "," $7}'
} >"$results_dir/table.csv"
