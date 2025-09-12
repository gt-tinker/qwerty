The structure of this project follows below. For each file, relevant sections
of the CGO '25 paper are written in bold in square brackets.

 * `qwerty/`: Python `qwerty` module, which also contains the code for the
   [Python C++ extension][3]
   * Python runtime:
     * `runtime.py`: Everything for the Python Qwerty runtime _except_ anything
       directly related to JIT compilation or `@qpu`/`@classical` kernels. This
       includes:
       * Python types defined to make the Python interpreter happy with our
         type annotations
       * Operator overloads to get `@classical` kernels to execute classically
       * Python implementations of histogram and continued fraction stuff
     * `err.py`: A rather elaborate trick for displaying a familiar-looking
       Python traceback for e.g. type checking errors in Qwerty kernels. Please
       see the giant comment in `qwerty/err.py` for details
     * `jit.py`: In charge of parsing Python source into ASTs and invoking the
       Qwerty JIT compiler when all captures are ready. Contains definitions of
       `@qpu` and `@classical` decorators. When a `@qpu` kernel is
       `__call__()`ed, this jumps into the JIT'd code via `_qwerty_harness`
       (below)
     * `convert_ast.py` **[Section 4]**: Converts a Python AST parsed with the
       [`ast` module][1] into a C++ Qwerty AST using the Python wrapper types
       exposed by `_qwerty_harness` (below)
   * Qwerty JIT compiler (C++):
     * `defs.hpp`: Forward declarations and misc. declarations factored out of
       `ast.hpp` to prevent a `#include` dependency cycle
     * `ast.{cpp,hpp}`: All Qwerty types and AST nodes (for both `@qpu` and
       `@classical` kernels)
     * `_qwerty_harness.cpp`: This is the interface between Python and C++.
       Qwerty AST nodes are exposed to Python to allow Python code to build a
       Qwerty AST and to call methods of AST nodes. (For a `QpuKernel` AST
       node, these methods include `compile()` to typecheck/JIT compile and
       `call()` to jump into the JIT'd code.)
     * `ast_visitor.hpp`: Declarations for Qwerty AST visitors, which evaluate
       dimension variable expressions, perform type checking, lower to MLIR, etc.
     * `ast_visitor/`: Definitions for Qwerty AST visitors declared in
       `ast_visitor.hpp`
       * `canonicalize.cpp` **[Section 4.2]**: Simplify the AST to facilitate
         producing higher-quality MLIR later.
       * `flag_dyn_basis.cpp`: Require that expressions of type `Basis` are
         only seen in specific syntactically valid places (not, e.g., as an
         argument to a function)
       * `graphviz.cpp`: Generate a graphviz `.dot` file for a Qwerty AST.
         Useful for debugging. [See `docs/debugging.md`.](debugging.md)
       * `eval_dimvar_exprs.cpp` **[Section 4]**: Given definitions of
         dimension variables, expand all dimension variable expressions. For
         example, if `N=3`, this visitor expands `'01'[N]` to `'01'+'01'+'01'`.
         A Repeat construct like `... | (f[[i]] for i in range(N)) | ...` would
         be expanded to `... | f[[0]] | f[[1]] | f[[2]] | ...`
       * `find_instants.cpp`: As part of `@qpu` type checking, gather all
         instantiations `f[[...]]` and materialize an instance of the kernel
         `f`, typecheck, and JIT it
       * `type_checking.cpp` **[Sections 4 and 4.1]**: Type checking for `@qpu`
         and `@classical` ASTs. This includes span equivalence checking for
         basis translations.
       * `qpu_lowering.cpp` **[Section 5.1]**: Lowering `@qpu` ASTs to MLIR
       * `classical_netlist.cpp`: Lowering `@classical` ASTs to [mockturtle][4]
         netlists
     * `mlir_handle.{cpp,hpp}` **[Sections 5.4 and 6.5]**: Wrapper around
       MLIRContext. Used by C++ code for constructing IR and by Python to kick
       off JITing. Also contains definitions of pass pipelines used for
       compilation. Instantiated by the Python runtime in `jit.py`
     * `tweedledum.cpp` **[Section 6.4]**: Responsible for transpiling
       [tweedledum][5] IR (`Circuit`s) to our MLIR qcirc dialect. This is
       called by both the frontend and the MLIR layer (since the MLIR layer
       uses tweedledum to synthesize the binary permutation at the heart of a
       basis translation).
     * `classical_kernel.cpp` **[Section 6.4]**: Compiles `@classical` kernels
       to the qcirc MLIR dialect (via Tweedledum) and instantiate them with
       `f.xor` etc
     * `qpu_kernel.cpp` **[Sections 5.1 and 7]**: Lowering Qwerty ASTs to the
       Qwerty MLIR dialect and glue code for invoking `@qpu` kernels (via
       QIR-EE or qir-runner)
 * `qwerty_mlir/`: Our MLIR dialects and passes for Qwerty
   * `Qwerty/`: An MLIR dialect that closely resembles Qwerty source code. The
     paper submission calls this dialect "Bases IR."
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
     [SSA/value semantics][6] rather than address semantics as QIR does
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
         peephole optimizations due to Liu et al.][8]
       * `DecomposeMultiControlPass.cpp` **[Section 6.5]**: Manually lowers
         multi-controlled gates using [Selinger's technique][7] to reduce T
         counts
       * `BaseProfilePrepPasses.cpp` **[Section 7]**: Transforms IR to prepare
         to lower to the QIR base profile
       * `QCircToQIRConversionPass.cpp` **[Section 7]**: Converts QCirc dialect
         to QIR (or more accurately, into the MLIR dialect named `llvm` with
         calls to QIR intrinsics)
 * `test/`: Unit tests. See [`docs/testing.md`](testing.md)
 * `eval/` **[Section 8]**: Evaluation scripts. See [`docs/eval.md`](eval.md)
 * `examples/`: Example Qwerty programs. See [`docs/examples.md`](examples.md)
   for more details.
 * `tpls/`: Third-party libraries, vendored here for posterity.

[1]: https://docs.python.org/3/library/ast.html
[3]: https://docs.python.org/3/extending/extending.html
[4]: https://github.com/lsils/mockturtle
[5]: https://github.com/boschmitt/tweedledum
[6]: https://doi.org/10.1145/3491247
[7]: https://arxiv.org/abs/1210.0974
[8]: https://doi.org/10.1109/CGO51591.2021.9370310
