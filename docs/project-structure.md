Project Structure
=================

This repository consists of the following five top-level subprojects:

1. `qwerty_ast` (Rust): Defines the Qwerty AST and typechecking/optimizations on it
2. `qwerty_mlir` (C++/Tablegen): MLIR dialects/passes for optimizing Qwerty
   programs and producing OpenQASM 3 or QIR
3. `qwerty_ast_to_mlir` (Rust): Converts a Qwerty AST to MLIR and JITs and runs it
4. `qwerty_util` (C++): C++ utility code, presently just a wrapper around
   [`tweedledum`][4]
5. `qwerty_pyrt` (Python/Rust): Defines the `qwerty` module, a little bit of Python that
   interfaces with the Rust code above via [PyO3][3]

There are also the following forks of third-party libraries that referenced as
git submodules:

1. [`qir_runner`][5] (Rust): Used for its implementation of the QIR runtime, which
   includes a good quantum simulator
2. [`tweedledum`][4] (C++): Used for synthesizes classical circuits (or classical
   permutations) as quantum circuits
3. `qwerty_mlir_sys` (Rust): A fork of [`mlir_sys`][1] that provides Rust
   bindings for the C API for MLIR dialects (both for our dialects and for
   upstream dialects)
4. `qwerty_melior` (Rust): A fork of [`melior`][2] a convenient wrapper for
   using MLIR APIs in Rust
5. `tblgen_rs` (Rust): A fork of [`tblgen_rs`][6], Rust bindings for
   [Tablegen][7] required by `melior` with no changes except upgrading LLVM.

The following sections describe the structure of these subprojects. For each
file, relevant sections of [the CGO '25 paper][8] are written in bold in square
brackets.

`qwerty_ast`: Qwerty Abstract Syntax Tree (Rust)
------------------------------------------------

 * `ast.rs` **[Sections 4 and 4.2]**: Defines the _plain_ Qwerty abstract
   syntax tree (AST) and how it should be [canonicalized][9]. Because this is
   the data structure on which type checking is performed, this plain AST does
   _not_ contain metaQwerty features such as qubit symbols (`'0'` becomes
   `__SYM_STD0__()` instead) or macros (`b.measure` becomes `__MEASURE__(b)`
   instead).
   * `ast/classical.rs`: AST nodes for `@classical` functions
   * `ast/qpu.rs`: AST nodes for `@qpu` kernels, including bases
 * `typecheck.rs` **[Sections 4 and 4.1]**: All type checking code, including
   checks for vector orthogonality for `bv1 + bv2` and basis span equivalence
   for `b1 >> b2`.
 * `meta.rs` **["AST expansion" heading in Section 4]**: Code that defines the
   metaQwerty AST and lowers it to a plain Qwerty AST.
   * `expand.rs`: Expands metaQwerty-specific constructs in the metaQwerty AST
     into constructs compatible with the plain AST.
   * `infer.rs`: Infers types and dimension variables from a metaQwerty AST.
   * `lower.rs`: Repeats expansion and inference to lower to a plain Qwerty
     AST.
 * `repl.rs`: Interprets Qwerty code as needed by the Qwerty Read–Eval–Print
   Loop (REPL).

`qwerty_mlir`: MLIR Dialects and Passes (C++)
---------------------------------------------

 * `Qwerty/`: An MLIR dialect whose semantics closely matches Qwerty semantics.
   The paper submission calls this dialect "Bases IR."
   * `IR/`:
     * `QwertyOps.td` **[Section 5]**: Definition of ops (instructions) in
       the Qwerty dialect
     * `QwertyAttributes.td` **[Section 5]**: Definition of attributes
       (compile-time metadata) in the Qwerty dialect, such as bases
     * `QwertyTypes.td` **[Section 5]**: Definition of types the Qwerty
       dialect, such as `qbundle[N]`
     * `QwertyInterfaces.td` **[Section 5.3]**: Contains the definition
       of the `Predicatable` op interface
     * `QwertyDialect.cpp` **[Sections 5.2-5.4]**: Contains
       `QwertyInlinerInterface`, which teaches MLIR how to inline calls to
       adjointed/predicated forms of functions
     * `QwertyOps.cpp` **[Section 5.4]**: Canonicalization patterns to
       simplify Qwerty IR (including converting indirect calls to direct
       calls)
   * `Utils/PredBlock.cpp` **[Section 5.3]**: Predicates a basic block
     (e.g., a function body)
   * `Analysis/`:
     * `QubitIndexAnalysis.{cpp,h}` **[Section 5.3]**: Performs analysis of
       how qubits flow through a basic block by assigning them indices.
       Ultimately used to infer swaps achieved by qubit renaming
     * `FuncSpecAnalysis.{cpp,h}` **[Section 6.2]**: Performs analysis on an
       entire MLIR ModuleOp to determine what specializations of what
       functions should be generated
   * `Transforms/`:
     * `LiftLambdasPass.cpp` **[Section 5.4]**: Converts `qwerty.lambda` ops
       into their own `qwerty.func` ops and replaces the SSA result with a
       `qwerty.func_const` op
     * `OnlyPredOnes.cpp`: Converts all predications into predicating only on
       $\vert 11 \cdots 1\rangle$
     * `QwertyToQCircConversionPass.cpp` **[Sections 6.1-6.3]**: Lowering
       Qwerty IR to QCirc IR. Performs circuit synthesis for basis
       translations and generates function specializations
 * `QCirc/`: A gate-level MLIR dialect. Maps closely to QIR, except it uses
   [SSA/value semantics][10] rather than address semantics as QIR does
   * `IR/`:
     * `QCircOps.td` **[Section 6]**: Definition of ops (instructions) in
       the QCirc dialect
     * `QCircTypes.td` **[Section 6]**: Definition of types the in QCirc
       dialect, such as `qubit`
   * `Utils/`:
     * `AdjointBlock.cpp` **[Sections 5.2]**: Takes adjoint of basic
       block (e.g., a function body)
     * `GenerateQasm.cpp` **[Section 7]**: Transpiles QCirc IR to OpenQASM 3
   * `Transforms/`:
     * `PeepholeOptimizationPass.cpp` **[Section 6.5]**: Performs common
       gate-level peephole optimizations as well as some basic [relaxed
       peephole optimizations due to Liu et al.][11]
     * `DecomposeMultiControlPass.cpp` **[Section 6.5]**: Manually lowers
       multi-controlled gates using [Selinger's trick][12] to reduce T
       counts
     * `BaseProfilePrepPasses.cpp` **[Section 7]**: Transforms IR to prepare
       to lower to the QIR base profile
     * `QCircToQIRConversionPass.cpp` **[Section 7]**: Converts QCirc dialect
       to QIR (or more accurately, into the MLIR dialect named `llvm` with
       calls to QIR intrinsics)
 * `CCirc/`: A simple MLIR dialect for classical digital logic circuits.
   Heavily inspired by the XAG (**X**OR–**A**ND **G**raph) in [mockturtle][13].
   * `IR/`:
     * `CCircOps.td`: Definition of ops (instructions) in
       the QCirc dialect
     * `QCircTypes.td`: Definition of types the in QCirc dialect, currently
       just `wire[N]`
 * `CAPI/`: C API for our MLIR dialects. Exposed as unsafe Rust by
   `qwerty_mlir_sys`, which `qwerty_melior` uses to provide a safe Rust API.

`qwerty_ast_to_mlir`: Lowering Qwerty ASTs to MLIR (Rust)
---------------------------------------------------------

* `ctx.rs`: Sets up the `MLIRContext`, including registering dialects
* `lower*.rs`: Lowers Qwerty AST nodes to MLIR, re-typechecking the AST in the
  process
* `jit.rs`: Runs passes on an `mlir::Module` and then JITs and invokes the
  code, where QIR intrinsics are bound to `qir_runner` implementations

`qwerty_util`: Tweedledum Adaptor (C++)
---------------------------------------

 * `util.hpp`: Macros useful throughout the compiler, such as the definition of
   pi and platform-agnostic bit manipulation macros
 * `tweedledum.{hpp,cpp}`: Converts a `ccirc.circuit` or a classical
   permutation to a quantum circuit using [tweedledum][4]

`qwerty_pyrt`: Python Runtime (Rust & Python)
---------------------------------------------

 * `python/` (Python): The Python runtime
   * `runtime.py`: All the APIs for the Python runtime for Qwerty _except_
     anything directly related to JIT compilation or `@qpu`/`@classical`
     kernels. This includes:
     * Python types (e.g., `qfunc` or `qubit`) defined to make the Python
       interpreter happy with our type annotations
     * Operator overloads on the `bit` type to get `@classical` kernels to
       execute classically
     * Python implementations of the measurement histogram and continued
       fraction algorithms
   * `err.py`: An elaborate trick for displaying a familiar-looking Python
     traceback for e.g. type checking errors in Qwerty kernels. Please
     see the giant comment in `err.py` for details
   * `kernel.py`: In charge of parsing Python source into ASTs and invoking the
     Qwerty JIT compiler when all captures are ready. Contains definitions of
     `@qpu` and `@classical` decorators. When a `@qpu` kernel is
     `__call__()`ed, this jumps into the JIT'd code via `qwerty_ast_to_mlir`
   * `convert_ast.py` **["AST generation" heading in Section 4]**: Converts a
     Python AST parsed with the [`ast` module][14] into a Qwerty AST using
     the Python wrapper types exposed by `_qwerty_pyrt` (the PyO3 Rust module
     described below)
   * `default_qpu_prelude.py`: Defines the default metaQwerty prelude,
     i.e, the statements that are preprended to every `@qpu` kernel
     automatically, such as macro definitions
   * `repl.py`: Top-level logic for the Qwerty REPL: printing the prompt,
     reading input, calling into Rust for evaluation, and printing the result
   * `__main__.py`: Invokes `repl.py`, allowing you to launch the Qwerty REPL
     by running `python -m qwerty`
 * `rust/`: The `_qwerty_pyrt` Python module defined in Rust using [PyO3][3]
   * `wrap_ast.rs`: Wraps the Rust data structures that make up the metaQwerty
     AST with Python classes, allowing Python code in `convert_ast` (see above)
     to create metaQwerty ASTs
   * `wrap_repl.rs`: Similar to above except for data structures for the Qwerty
     REPL
   * `lib.rs`: Registers all Python wrapper objects

[1]: https://github.com/mlir-rs/mlir-sys/
[2]: https://github.com/mlir-rs/melior/
[3]: https://pyo3.rs/
[4]: https://github.com/boschmitt/tweedledum
[5]: https://github.com/qir-alliance/qir-runner/
[6]: https://github.com/mlir-rs/tblgen-rs/
[7]: https://llvm.org/docs/TableGen/
[8]: https://dl.acm.org/doi/10.1145/3696443.3708966
[9]: https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html
[10]: https://doi.org/10.1145/3491247
[11]: https://doi.org/10.1109/CGO51591.2021.9370310
[12]: https://arxiv.org/abs/1210.0974
[13]: https://github.com/lsils/mockturtle
[14]: https://docs.python.org/3/library/ast.html
