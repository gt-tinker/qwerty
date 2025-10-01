Upgrading LLVM
==============

Upgrading LLVM often provides bugfixes and improved APIs.

Step 1: Building LLVM
---------------------

The first step of upgrading LLVM is building it. To do this, create a new
release named "LLVM XX.Y.Z" and with the tag `vXX.Y.Z` [in the `gt-tinker/qwerty-llvm-builds` repository][4]. GitHub Actions should build the tarballs automatically; you can track the [progress under the Actions tab][5].

Step 2: Replacing Occurrences of the Old Version
------------------------------------------------

The next step is replacing every occurrence of `llvm21`, `21_1_1`, and `21.1.1`
throughout this repository. This command is useful for this task (but make sure
to check the results are right with `git diff`):

    $ git sed 's/\<llvm21\>/llvmXX/g ; s/v21_1_1/vXX_Y_Z/g ; s/\<21_1_1\>/XX_Y_Z/g ; s/21\.1\.1/XX.Y.Z/g'

The `git sed` command used above is a custom alias. To use it, you need to
define the alias yourself. Here’s how to set it up:

For macOS:

    $ git config --global alias.sed '! git ls-files -s | grep -v '\''^16'\'' | cut -f 2- | xargs env LC_ALL=C sed -E -i '\'\'' -e'

For Linux:

    $ git config --global alias.sed '! git ls-files -s -z | grep -z -v '\''^16'\'' | cut -z -f 2- | xargs -0 sed -i -e'

Step 3: Testing MLIR Build
--------------------------

The next step is testing to see if `qwerty_mlir` builds correctly. See the
steps in [`build.md`](build.md). Try the `FileCheck` tests.

Step 4: Upgrading Rust Libraries
--------------------------------

There are two possiblities:

 * If it's a major version upgrade (e.g., LLVM 21 to LLVM 22) the following
   code changes are needed:
   1. `qwerty_mlir_sys`: Fix `LLVM_MAJOR_VERSION` in `build.rs` and
      `package.version` in `Cargo.toml` as seen in [this example PR][1].
   2. `tblgen_rs`: Update `record_keeper.rs`, `build.rs`, and `Cargo.toml` as
       seen in [this example PR][2].
   3. `qwerty_melior`: Update `LLVM_MAJOR_VERSION` in `macro/build.rs` and the
      `tblgen` feature in `macro/Cargo.toml` as seen in [this example PR][3].
   It is polite to send these changes upstream if you can.
 * Else, no code changes are needed

Step 5: End-To-End Testing
--------------------------

As a final check, rebuild the Python runtime and run the whole test suite
(`dev/run-tests.sh`). See [the README](../README.md).

[1]: https://github.com/mlir-rs/mlir-sys/pull/71/files
[2]: https://github.com/mlir-rs/tblgen-rs/pull/31/files
[3]: https://github.com/mlir-rs/melior/pull/731/files
[4]: https://github.com/gt-tinker/qwerty-llvm-builds/releases
[5]: https://github.com/gt-tinker/qwerty-llvm-builds/actions/workflows/build-llvm.yml
