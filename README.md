The Compiler for Qwerty, a Basis-Oriented Quantum Programming Language
======================================================================

This is the compiler (and examples) for Qwerty, a quantum programming language
embedded in Python. It is licensed under the MIT license. If you want to contribute
or have issues, you can check out [`CONTRIBUTING.md`](CONTRIBUTING.md).

Documentation
-------------

The `docs/` directory contains more documentation that does not fit in this
README:

 * Useful for newcomers:
   * [`docs/examples.md`](docs/examples.md): A list of the Qwerty example
     programs available in the `examples/` directory
   * [`docs/project-structure.md`](docs/project-structure.md): An overview
     of the contents of this project (i.e., which files do what) and how the
     sections in the paper submission map to source files
 * Compiler development/maintenance guides:
   * [`docs/build.md`](docs/build.md): Instructions for building subsets of the
     Qwerty compiler when working on particular components
   * [`docs/testing.md`](docs/testing.md): Details on our multi-faceted testing
     framework
   * [`docs/wsl-tips.md`](docs/wsl-tips.md): Recommendations for building with
     WSL on Windows
   * [`docs/debugging.md`](docs/debugging.md): Tricks for debugging the Qwerty
     compiler
   * [`docs/profiling.md`](docs/profiling.md): Shows how to profile the Qwerty
     compiler itself
   * [`docs/build-llvm.md`](docs/build-llvm.md): Steps to build LLVM yourself
   * [`docs/upgrading-llvm.md`](docs/upgrading-llvm.md): Describes the
     semi-automated process for upgrading the version of LLVM used by the Qwerty
     compiler
 * Integration with other tools:
   * [`docs/qiree.md`](docs/qiree.md): Shows how to build the Qwerty compiler
     with [QIR-EE][4] support
 * Compiler code documentation:
   * [`docs/new-mlir-attr-rust.md`](docs/new-mlir-attr-rust.md): Demonstrates the
     interplay betwen Tablegen, C++, C, and Rust needed to add a new attribute to
     one of our MLIR dialects
   * [`docs/state-prep.md`](docs/state-prep.md): Describes the arbitrary state
     preparation technique by Shende et al. that we use to synthesize circuits for
     superposition literals

The rest of this README is dedicated to installation, basic testing, and
troubleshooting.

Getting Started
---------------

This guide describes the common case of building the full Qwerty
compiler/runtime. (For instructions on building some portions of the compiler
independently, see [`docs/build.md`](docs/build.md).)

### Dependencies

You need to install [Rust][1] _and_ the following:

1. LLVM 21.1.1 (with MLIR). First, download the LLVM build archive that is
   appropriate for your OS and architecture [from our repository][2]. (If you
   want to build LLVM yourself, see [this guide](docs/build-llvm.md).) Then you
   need to set both of the following environment variables (assuming
   `$HOME/bin` is where you extracted the LLVM archive, for example):
   ```
   $ export PATH=$PATH:$HOME/bin/llvm21/bin/
   $ export MLIR_DIR=$HOME/bin/llvm21/lib/cmake/mlir/
   ```
   You should set these persistently, e.g., in your `~/.bashrc` or `~/.zshrc`.

2. The following packages are also needed to build:

   **Debian/Ubuntu:**
```
   $ sudo apt install build-essential cmake ninja-build zlib1g-dev libclang-dev
```
**macOS (Homebrew):**
```
   $ xcode-select --install
   $ brew install cmake ninja zlib llvm
```
If you are on macOS, the XCode developer tools (`xcode-select --install`) are required in place of `build-essential`. You may also need to export the Homebrew LLVM path:
```
   $ export PATH="$(brew --prefix llvm)/bin:$PATH"
```

### Building

To generate the Python extension, run the following:

    $ git submodule update --init
    $ python3 -m venv venv
    $ . venv/bin/activate
    $ cd qwerty_pyrt
    $ pip install maturin numpy
    $ maturin develop -vv

The last command will (re)build _everything_. If you need additional verbosity for troubleshooting, passing `-vvv` to
`maturin develop` helps to give you a better idea of what is going on.

### Testing

As a quick smoke test, you can run a single Qwerty program from `examples/` as
follows:

    $ python ../examples/deutsch.py
    Balanced f:
    Classical: f(0) xor f(1) = 1
    Quantum:   f(0) xor f(1) = 1

    Constant f:
    Classical: f(0) xor f(1) = 0
    Quantum:   f(0) xor f(1) = 0

To run _all_ tests across the compiler, run the following:

    $ ../dev/run-tests.sh

Troubleshooting WSL Issues
--------------------------

If you are working on Windows, we recommend building with [Windows Subsystem
for Linux (WSL)][5]. However, the default (low) memory allocation for WSL and
slow access to Windows filesystems can cause issues. Please see
[`docs/wsl-tips.md`](docs/wsl-tips.md) for more information.

Citation
--------

To cite the Qwerty compiler, please cite our CGO '25 paper:

Austin J. Adams, Sharjeel Khan, Arjun S. Bhamra, Ryan R. Abusaada, Anthony M.
Cabrera, Cameron C. Hoechst, Travis S. Humble, Jeffrey S. Young, and Thomas M.
Conte. March 2025. **"ASDF: A Compiler for Qwerty, a Basis-Oriented Quantum
Programming Language."** In _Proceedings of the 23rd ACM/IEEE International
Symposium on Code Generation and Optimization (CGO '25)_.
https://doi.org/10.1145/3696443.3708966

BibTeX citation:

    @inproceedings{adams_asdf_2025,
        author = {Adams, Austin J. and Khan, Sharjeel and Bhamra, Arjun S. and Abusaada, Ryan R. and Cabrera, Anthony M. and Hoechst, Cameron C. and Humble, Travis S. and Young, Jeffrey S. and Conte, Thomas M.},
        title = {ASDF: A Compiler for Qwerty, a Basis-Oriented Quantum Programming Language},
        year = {2025},
        isbn = {9798400712753},
        url = {https://doi.org/10.1145/3696443.3708966},
        doi = {10.1145/3696443.3708966},
        booktitle = {Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization},
        series = {CGO '25}
        pages = {444–458},
        numpages = {15},
    }


The evaluation for the CGO '25 paper uses additional code not on the `main`
branch of this repository. You can find that evaluation code in [the Zenodo
artifact][3] or on the `cgo25-artifact` branch.

[1]: https://www.rust-lang.org/tools/install
[2]: https://github.com/gt-tinker/qwerty-llvm-builds/releases/tag/v21.1.1
[3]: https://doi.org/10.5281/zenodo.14080494
[4]: https://arxiv.org/abs/2404.14299
[5]: https://learn.microsoft.com/en-us/windows/wsl/about
