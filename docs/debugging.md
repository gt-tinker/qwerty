Debugging
=========

If you want debug symbols in the C++ extension (for e.g. gdb), change the last
two lines of `pyproject.toml` as follows:
```diff
-install.strip = true
-cmake.build-type = "Release"
+install.strip = false
+cmake.build-type = "Debug"
```

To see debugging info such as IR or ASTs, set the environment variable
`QWERTY_DEBUG=1`. The following files will be created in the current working
directory:

* [Python AST][5] for a kernel `k` in `mymodule.py` (`@qpu` or `@classical`):
  `mymodule_k.pyast`
* Qwerty AST for a kernel `k` in `mymodule.py` (`@qpu` or `@classical`):
  `mymodule_k.dot`
* MLIR for different stages of compilation: `module*.mlir` (and the LLVM IR
  `module*.ll` for QIR-EE, if enabled at build time)
* Classical netlist for a `@classical` function `f`: `mockturtle_net_f.dot`
* Tweedledum quantum circuit for a `@classical` function `f`:
  `tweedledum_circ_f.txt`
* If [built with QIR-EE][6], logging from QIR-EE/XACC in `qiree.log` and the
  [Base Profile QIR][1] in `module_qir.ll`

Example commands:

    $ cd examples
    $ QWERTY_DEBUG=1 python bv_driver.py 1101
    $ vim bv_*.pyast module_*.mlir tweedledum_circ_*.txt
    $ ./render.sh bv_kernel.dot # Writes Qwerty AST image to bv_kernel.png
    $ ./render.sh mockturtle_net_f.dot # Writes classical netlist to mockturtle_net_f.png

This project contains `qwerty-opt` for running MLIR passes. `pip` will
install it in your `$PATH`. You can run `qwerty-opt` like this:

    $ qwerty-opt --pre-lowering-cleanup --convert-qwerty-to-qcirc examples/module_qwerty_opt.mlir

Sometimes, however, it may be more convenient to debug a pass invoked from the
Qwerty runtime instead of `qwerty-opt`. If you want to see debug printouts from
inside LLVM ([example][3]), which you could normally enable with
`qwerty-opt -debug-only=inlining` [or `mlir-opt`][4], you can use the following
[hack][7]:

    llvm::DebugFlag = true;
    llvm::setCurrentDebugType("inlining");
    // ... call some LLVM/MLIR code ...
    llvm::DebugFlag = false;

The `"inlining"` string above is found from the definition of [the `DEBUG_TYPE`
macro in the LLVM/MLIR code][2]. (You can find more info about `LLVM_DEBUG` in
[the LLVM documentation][8].) **Note:** Seeing output requires `NDEBUG` _not_
to be set at LLVM/MLIR compile time, which actually requires passing the
`-DLLVM_ENABLE_ASSERTIONS=TRUE` CMake flag shown above at LLVM/MLIR build time,
unless you are doing a `CMAKE_BUILD_TYPE=Debug` build (but beware, Debug builds
of LLVM are gargantuan.)

Debugging MLIR in gdb
---------------------

If some built-in MLIR pass is segfaulting or not behaving as you expect,
stepping through the MLIR code in gdb may be necessary. To do this:

1. Download an LLVM build with debug symbols, [linked
   above](#option-1-use-a-pre-built-llvm-tarball), extract it, and set up
   environment variables (as [described above](#macoslinux))
2. Run the offending Qwerty program with `$QWERTY_DEBUG` set to `1` to generate
   some `module_*.mlir` files ([described above](#debugging))
3. Clone [the LLVM repository][7] locally and check out the tag
   `llvmorg-20.1.6`. This way, gdb can print out lines of code as you step
   through the MLIR/LLVM source code.
4. If you cloned LLVM at `$LLVM_REPO_PATH`, run qwerty-opt in gdb as follows:
   ```
   $ gdb -ex "set substitute-path /nethome/aadams80/USERSCRATCH/llvm $LLVM_REPO_PATH" --args qwerty-opt --debug-only=inlining --inline module_qwerty_opt.mlir
   ```
   The `substitute-path` part remaps the path to the directory where I built
   LLVM (which is baked into the binary) to your local copy of the LLVM source
   code.

[1]: https://github.com/qir-alliance/qir-spec/blob/8b3fd47b7b70122a104e24733ef9de911576f7d6/specification/under_development/profiles/Base_Profile.md
[2]: https://github.com/llvm/llvm-project/blob/ef33d6cbfc2dada0709783563fee57e6afd9a2d9/mlir/lib/Transforms/Inliner.cpp#L33
[3]: https://github.com/llvm/llvm-project/blob/ef33d6cbfc2dada0709783563fee57e6afd9a2d9/mlir/lib/Transforms/Utils/InliningUtils.cpp#L139-L142
[4]: https://mlir.llvm.org/getting_started/Debugging/
[5]: https://docs.python.org/3/library/ast.html
[6]: qiree.md
[7]: https://github.com/llvm/llvm-project
[8]: https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option
