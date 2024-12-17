#!/bin/bash
set -e
if [[ $# -lt 2 ]]; then
    printf 'usage: %s output.qasm file.hs args...\n' "$0" >&2
    exit 1
fi
out_file=$1
file=$2
shift 2
quipper "$file"
QasmPrinting -3 -inline <("./${file%.hs}" "$@") >"$out_file"
