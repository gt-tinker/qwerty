Debugging
=========

This document describes some common tricks useful in debugging the Qwerty
compiler.

Running an Integration Test in gdb
----------------------------------

To run a particular integration test in `gdb` (e.g., `bv_nometa`):

    $ cd qwerty_pyrt/
    $ gdb --args python3 -c 'import qwerty.tests.integ.nometa.bv; qwerty.tests.integ.nometa.bv.test(1)'

Dumping AST and MLIR
--------------------

To dump the Qwerty AST and MLIR for a Qwerty program, set the environment
variable `QWERTY_DEBUG=1`. The following will be created in the current working
directory:

* `qwerty-debug/`
  * `qwerty_ast.py`: The most recently executed plain Qwerty AST
  * `mlir/`: A directory holding the full MLIR before and after every pass

Example command to generate these debug files:

    $ cd examples/
    $ QWERTY_DEBUG=1 python bell.py
    The Qwerty AST file will be dumped to file `[snip]/examples/qwerty-debug/qwerty_ast.py`
    MLIR files will be dumped to directory `[snip]/examples/qwerty-debug/mlir`
    00 -> 50.39%
    11 -> 49.61%

To examine the plain Qwerty AST:

    $ cat qwerty-debug/qwerty_ast.py
    from qwerty import *

    @qpu(prelude=None)
    def kernel_0() -> bit[2]:
        id = ({__SYM_PAD__()}) >> ({__SYM_PAD__()})
        discard = __DISCARD__()
    [snip]

To run the plain Qwerty AST:

    $ python qwerty-debug/qwerty_ast.py
    00 -> 48.73%
    11 -> 51.27%

To examine the initial MLIR produced from the Qwerty AST:

    $ cat qwerty-debug/mlir/*/0_canonicalize.mlir
    // -----// IR Dump Before Canonicalizer (canonicalize) //----- //
    module {
      qwerty.func @kernel_0[]() irrev-> !qwerty<bitbundle[2]> {
        %0 = qwerty.lambda[](%arg0: !qwerty<qbundle[1]>) rev-> !qwerty<qbundle[1]> {
    [snip]

To examine the final LLVM-dialect MLIR:

    $ cat qwerty-debug/mlir/*/31_canonicalize.mlir
    // -----// IR Dump After Canonicalizer (canonicalize) //----- //
    module {
      llvm.func @kernel_0() -> !llvm.ptr attributes {llvm.emit_c_interface} {
        %0 = llvm.mlir.constant(1 : i64) : i64
        %1 = llvm.mlir.constant(0 : i64) : i64
    [snip]

To run a pass on the initial MLIR (e.g., the canonicalizer):

    $ ../dev/bin/qwerty-opt -canonicalize qwerty-debug/mlir/*/0_canonicalize.mlir
    module {
      qwerty.func @kernel_0[]() irrev-> !qwerty<bitbundle[2]> {
        %0 = qwerty.lambda[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<bitbundle[1]> {
          %4 = qwerty.qbmeas %arg0 by {list:{"|0>", "|1>"}} : !qwerty<qbundle[1]> -> !qwerty<bitbundle[1]>
    [snip]

To convert the final LLVM-dialect MLIR to LLVM IR:

    $ ../dev/bin/qwerty-translate -mlir-to-qir qwerty-debug/mlir/*/31_canonicalize.mlir
    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"

    define ptr @kernel_0() {
      %1 = call ptr @__quantum__rt__qubit_allocate()
      %2 = call ptr @__quantum__rt__qubit_allocate()
    [snip]

Debug Output from MLIR
----------------------

Sometimes, however, it may be more convenient to debug a pass invoked from the
Qwerty runtime instead of `qwerty-opt`. If you want to see debug printouts from
inside LLVM ([example][2]), which you could normally enable with
`qwerty-opt -debug-only=inlining` [or `mlir-opt`][3], you can use the following
[hack][4]:

    llvm::DebugFlag = true;
    llvm::setCurrentDebugType("inlining");
    // ... call some LLVM/MLIR code ...
    llvm::DebugFlag = false;

The `"inlining"` string above is found from the definition of [the `DEBUG_TYPE`
macro in the LLVM/MLIR code][1]. (You can find more info about `LLVM_DEBUG` in
[the LLVM documentation][5].) **Note:** Seeing output requires `NDEBUG` _not_
to be set at LLVM/MLIR compile time, which actually requires passing the
`-DLLVM_ENABLE_ASSERTIONS=TRUE` CMake flag shown above at LLVM/MLIR build time,
unless you are doing a `CMAKE_BUILD_TYPE=Debug` build (but beware, Debug builds
of LLVM are gargantuan.)

Debugging MLIR in gdb
---------------------

If some built-in MLIR pass is segfaulting or not behaving as you expect,
stepping through the MLIR code in gdb may be necessary. To do this:

1. You will need to build LLVM with debug symbols. See
   [`build-llvm.md`](build-llvm.md) (use `CMAKE_BUILD_TYPE=RelWithDebInfo` or `Debug`).
2. Run the offending Qwerty program with `$QWERTY_DEBUG` set to `1` to generate
   some `module_*.mlir` files ([described above](#debugging))
3. Clone [the LLVM repository][4] locally and check out the tag
   `llvmorg-21.1.1`. This way, gdb can print out lines of code as you step
   through the MLIR/LLVM source code.
4. If you you built LLVM on your machine, you can run `qwerty-opt` in gdb as usual:
   ```
   $ gdb --args dev/bin/qwerty-opt --debug-only=inlining --inline module_qwerty_opt.mlir
   ```
   But if you built LLVM on a different machine, clone LLVM on your local
   machine `$LLVM_REPO_PATH` and run `qwerty-opt` in gdb as follows:
   ```
   $ gdb -ex "set substitute-path /nethome/aadams80/USERSCRATCH/llvm $LLVM_REPO_PATH" --args dev/bin/qwerty-opt --debug-only=inlining --inline module_qwerty_opt.mlir
   ```
   The `substitute-path` part remaps the path to the directory where I built
   LLVM (which is baked into the binary) to your local copy of the LLVM source
   code.

[1]: https://github.com/llvm/llvm-project/blob/ef33d6cbfc2dada0709783563fee57e6afd9a2d9/mlir/lib/Transforms/Inliner.cpp#L33
[2]: https://github.com/llvm/llvm-project/blob/ef33d6cbfc2dada0709783563fee57e6afd9a2d9/mlir/lib/Transforms/Utils/InliningUtils.cpp#L139-L142
[3]: https://mlir.llvm.org/getting_started/Debugging/
[4]: https://github.com/llvm/llvm-project
[5]: https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option
