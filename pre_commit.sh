#!/bin/bash

# List of submodules to update
submodules=(
    "qir_runner"
    "qiree"
    "qsim"
    "qwerty_melior"
    "qwerty_mlir_sys"
    "tblgen_rs"
    "tweedledum"
)

# Loop through each submodule and pull it from main
for mod in "${submodules[@]}"; do
    echo "Pulling $mod from main..."
    git checkout main -- "$mod"
done

echo "All submodules updated!"

