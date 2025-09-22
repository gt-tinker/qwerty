#ifndef DIALECT_INCLUDE_QCIRC_SYNTH_QCIRC_SYNTH_H
#define DIALECT_INCLUDE_QCIRC_SYNTH_QCIRC_SYNTH_H

#include "CCirc/IR/CCircOps.h"

// Circuit synthesis routines for the QCirc dialect

namespace qcirc {

// Synthesize a classical permutation as QCirc IR, inserting QCirc ops inline
// wherever the builder's insertion point is, without creating a new FuncOp.
// The `control_qubits` argument adds additional controls to each gate.
// Per the tweedledum gods, `perm` means the following:
// > A permutation is specified as a vector of `2^n` different integers
// > ranging from `0` to `2^n-1`.
// This function is called "slow" because `perm` is exponentially large and
// this algorithm runs in exponential time (hence "slow").
void synthPermutationSlow(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx,
        const std::vector<uint32_t> &perm);

// Synthesize a classical permutation as QCirc IR, inserting QCirc ops inline
// wherever the builder's insertion point is, without creating a new FuncOp.
// The `control_qubits` argument adds additional controls to each gate.
// Per the tweedledum gods, `perm` means the following:
// > A permutation is specified as a vector of `2^n` different integers
// > ranging from `0` to `2^n-1`.
// This function runs in polynomial time, but it will not produce as
// high-quality of a circuit as `synthPermutationSlow()`.
void synthPermutationFast(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx,
        const llvm::SmallVector<std::pair<llvm::APInt, llvm::APInt>> &perm);

void synthBennettFromXAG(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        ccirc::CircuitOp xag_circ,
        llvm::SmallVectorImpl<mlir::Value> &qubits);

} // namespace qcirc

#endif // DIALECT_INCLUDE_QCIRC_SYNTH_QCIRC_SYNTH_H
