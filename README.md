# Creating a Qwerty AST in Rust

We want to leverage some of Rust's type features, pattern matching, and existing infrastructure (see: `pyo3`, `maturin`, `melior`, etc.) to streamline the type checking, etc. This is the first step in making that happen.

## File structure:
- `main.rs`: This file is currently full of tests that check the functionality of the AST structure, as well as some of the type inference and dimension variable constraint solving functionality
- `ast.rs`: The `ast` file defines the AST Enum-Struct, as well as all of the relevant impl blocks for helper functions. The latter half is just struct definitions for each of the major AST nodes/major Qwerty types and invariants.
- `basis.rs`: In trying to make the delineation clearer between bases and all other Qwerty types, this file seeks to only define the relevant basis elements, building up bases from `QubitSymbol`s to `BasisVector`s, etc.
- `types.rs`: Here we define types according to the Mini-Qwerty specification, provided in the appendix of the pre-print paper [here](https://arxiv.org/pdf/2404.12603). See the paper for more exact details.
- `dimexpr.rs`: Here we define the dimension variable expressions, such as `2*(N+5)`, which are used to define constraints/bounds on (qu)bit registers in Qwerty. In addition to the basic definition of these types, this file also contains:
    1. A parser for dimension variable expressions, to make them easier to work with, *and*
    2. A series of solvers for simple algebraic rules, such as the associative and distributive properties. It is rudimentary but generally simplifies to a useful form that we can work with later.
- `inference.rs`: This file defines all of the necessary processing to do a (currently limited) form of Hindley-Milner type inference, as well as the runner for the dimvar expr solver; a good article on type inference/reconstruction can be found [here](https://course.ccs.neu.edu/cs4410sp19/lec_type-inference_notes.html).
The type inference follows the above link closely, but has a notable feature missing right now. As an example, take the following piped commands in a Qwerty program:

```python
def some_func(ins) -> outs:
    '0'[N] | id[M+2] | flip[5]
```
Within this function, the `id` function is currently specified as having type `Qubit[5] -> Qubit[5]` after type reconstruction. This definition will then apply anywhere else the `id` is used, even if the next register `id` is applied to doesn't have 5 qubits. Instead, what we need is a type *scheme* instead of just type variables: a type scheme allows us to genericize a function during type reconstruction by providing a type variable that the function can be generic over! Think of it as a "for all" quantifier, so `id` could be rewritten as "For all TypeBounds(X) (you'd generate this dynamically, similar to TypeVars), `id` has type `Qubit[X] -> Qubit[X]`". This guarantees that you can have an `id` specified to qubit registers of any dimension.

The way the dimension variable expression solver works is as follows:
    1. While traversing the AST, within function nodes, keep track of all dimension variable expressions, and encode them into the function's `ConstraintInfo` struct.
    2. After we have recursed through the function's AST nodes, generate constraints based on the stored `DimExpr`s, 
    3. (Within the `solve` function) Simplify two candidate constraints and move all constant values to one side, then create a matrix representing a system of equations for our constraints and solve it, yielding values for our dimension variables.

## Coverage
To get coverage, run:
```
$ cargo llvm-cov --html
```
Then you can open `qwerty_ast/target/llvm-cov/html/index.html` (relative to the
repo root) in your browser.

You may need to install `cargo-llvm-cov` first with:
```
$ cargo +stable install cargo-llvm-cov --locked
```

# Qwerty MLIR dialects

To build the MLIR code in `qwerty_mlir` (and the other C++ code in
`qwerty_util`), make sure `llvm-config` is in your `$PATH`. Then run:

    $ git submodule update --init tpls/tweedledum
    $ mkdir build && cd build
    $ cmake -G Ninja ..
    $ ninja
