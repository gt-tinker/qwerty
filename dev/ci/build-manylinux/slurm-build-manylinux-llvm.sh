#!/bin/bash
#SBATCH -Jqwerty-llvm-ci        # Job name
#SBATCH -N1 --cpus-per-task=32  # Number of nodes and CPUs per node required
#SBATCH --mem-per-cpu=2G        # Memory per core
#SBATCH -t 03:00:00             # Duration of the job (3 hours)
#SBATCH -p rg-nextgen-hpc       # Partition Name
#SBATCH -o /tools/ci-reports/qwerty-manylinux-llvm-ci-%j.out   # Combined output and error messages file
#SBATCH -W                      # Do not exit until the submitted job terminates.

cd "$GITHUB_WORKSPACE"
hostname

cd dev/ci/build-manylinux
exec ./build-manylinux-llvm.sh apptainer "$LLVM_VERSION" /projects/ci-runners/qwerty-llvm-builds/llvm
