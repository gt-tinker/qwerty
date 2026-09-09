# Basis Translation Algorithm

## Overview

The QWERTY interpreter previously hard-coded quantum basis translations to specific outcomes. This document describes the general algorithm implemented to replace that approach with a correct, comprehensive basis translation procedure.

## Background

QWERTY is a quantum programming language designed for ease-of-use in both academic and industry settings. Its interpreter allows byte-sized quantum programs to be run interactively (e.g., in Jupyter-style cells). A key operation in the interpreter is **basis translation**: given two quantum bases, produce a unitary matrix that maps one to the other.

## Algorithm

The algorithm takes two quantum bases as input:

- `b1 = {bv₁, bv₂, ..., bvₙ}` — the input basis (AST nodes)
- `b2 = {bv₁', bv₂', ..., bvₙ'}` — the output basis (AST nodes)

It produces a `2ⁿ × 2ⁿ` unitary matrix `U_total` that performs the map `b1 → b2`.

Each basis vector AST node is first converted to its dense vector representation using the recursive tensor-product procedure defined in Figure 12 of the [QWERTY paper](https://arxiv.org/abs/2404.12603) (Appendix A).

### Step 1 — Build the cross-basis map `M`

Compute pairwise outer products between the output basis vectors (as kets) and the input basis vectors (as bras):

$$M = \sum_i |bv_i'\rangle\langle bv_i|$$

Concretely, `M` is a matrix whose columns correspond to input basis vectors and whose rows correspond to output basis vectors:

$$M = \begin{pmatrix} & |bv_1\rangle & |bv_2\rangle & \cdots & |bv_n\rangle & |bv_j^\perp\rangle \\ |bv_1'\rangle & 1 & 0 & \cdots & 0 & 0 \\ |bv_2'\rangle & 0 & 1 & \cdots & 0 & 0 \\ \vdots & 0 & 0 & \ddots & 0 & 0 \\ |bv_n'\rangle & 0 & 0 & \cdots & 1 & 0 \\ |bv_j^\perp\rangle & 0 & 0 & \cdots & 0 & 0 \end{pmatrix}$$

where `|bvⱼ⊥⟩` spans `span(b1)⊥`.

The null space of `M` is the orthogonal complement of the input basis span:

$$\text{null}(M) = \text{span}(b1)^\perp$$

### Step 2 — Build the complement projector `P_U`

Define `W = span(b1)`. The projector onto `W` is:

$$P_W = \sum_i |bv_i\rangle\langle bv_i|$$

This reuses the same outer-product routine from Step 1, just self-paired (`|bvᵢ⟩⟨bvᵢ|`) rather than cross-paired (`|bvᵢ'⟩⟨bvᵢ|`), which makes both computable from a single shared subroutine.

The projector onto the orthogonal complement is then:

$$P_U = I - P_W$$

> **On orthonormality:** In a general mathematical setting, orthonormality of the input/output bases cannot be assumed. However, QWERTY's type checker guarantees at compile time that all quantum basis expressions are well-typed, which entails orthonormality. The algorithm therefore relies on this guarantee rather than re-verifying it at runtime.

### Step 3 — Combine into the total unitary

Since `M` and `P_U` are both `2ⁿ × 2ⁿ` matrices, the total basis translation operator is their sum:

$$U_{\text{total}} = M + P_U$$

### Step 4 — Send to the simulator

`U_total` is dispatched to the simulator for execution. Scheduling and output handling are managed externally; this step has no further synchronization requirements.

## Complexity and Future Work

This formulation opens the door to further optimizations, including fast matrix multiplication when forming `M` for large basis sets. The algorithm also improves usability for the broader class of QWERTY programs by removing the hardcoded translation constraints that previously limited the interpreter.

## Reference

Adams, A. J., Khan, S., Bhamra, A. S., Abusaada, R. R., Young, J. S., & Conte, T. M. (2025). *QWERTY: A Basis-Oriented Quantum Programming Language.* arXiv:2404.12603 [quant-ph]. https://arxiv.org/abs/2404.12603
