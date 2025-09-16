#ifndef DIALECT_INCLUDE_QCIRC_UTILS_UTILS_H
#define DIALECT_INCLUDE_QCIRC_UTILS_UTILS_H

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "CCirc/IR/CCircOps.h"

// Miscellaneous utilities for the QCirc dialect, specifically for generating
// OpenQASM 3.0 and taking the adjoint of basic blocks.

namespace qcirc {

// Generate OpenQASM 3.0 for the FuncOp passed. print_locs with print comments
// with mlir::Location (debug info). Result is returned via result.
mlir::LogicalResult generateQasm(mlir::func::FuncOp func_op,
                                 bool print_locs,
                                 std::string &result);

class Reshaper {
public:
    virtual ~Reshaper() {}
    // It is possible to write a reversible function that takes a qbundle[2]
    // and returns qbundle[1], qbundle[1]. It is the job of this class to take
    // the qbundle[2], unpack it, and repack as two single-qubit qbundles.
    virtual mlir::LogicalResult reshape(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange block_args,
        mlir::TypeRange types_needed,
        llvm::SmallVectorImpl<mlir::Value> &reshaped_out) = 0;
};

// A reshaper that always fails. This is realistically useful in the Qwerty
// compiler because reversible functions in the IR always take a qbundle[N] and
// return a qbundle[N], so no reshaping is needed. (However, we keep Reshaper
// around in case this changes in the future.)
class TrivialReshaper : public Reshaper {
public:
    virtual mlir::LogicalResult reshape(
            mlir::RewriterBase &rewriter,
            mlir::ValueRange block_args,
            mlir::TypeRange types_needed,
            llvm::SmallVectorImpl<mlir::Value> &reshaped_out) override {
        assert(0 && "Reshaping should never be needed in the QCirc dialect");
        return mlir::failure();
    }
};

// Take the adjoint of a basic block in-place except don't touch the block
// terminator — the would-be arguments to the new terminator are returned via
// term_args. The reshaper is used when the block arguments do not exactly
// match the terminator arguments (to reshape them, e.g., convert two
// qbundle[2]s into one qbundle[4]). The adjointing is accomplished with the
// Adjointable op interface.
mlir::LogicalResult takeAdjointOfBlockInPlaceNoTerm(
        mlir::RewriterBase &rewriter,
        Reshaper &reshaper,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &term_args);

// Same as takeAdjointOfBlockInPlaceNoTerm() above except pretends block_args
// are the arguments to the basic block. (Why? The MLIR inliner replaces block
// arguments with the call site arguments before it gives us a chance to
// process the inlined blocks. See QwertyDialect.cpp.)
mlir::LogicalResult takeAdjointOfBlockInPlaceNoTerm(
        mlir::RewriterBase &rewriter,
        Reshaper &reshaper,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        llvm::SmallVectorImpl<mlir::Value> &term_args);

// Same as takeAdjointOfBlockInPlaceNoTerm() except also re-creates the
// terminator (which will have op type TerminatorType) and assumes no reshaper
// is needed (and throws an error otherwise).
template<typename TerminatorType>
mlir::LogicalResult takeAdjointOfBlockInPlace(
        mlir::RewriterBase &rewriter,
        mlir::Block &fwd_block,
        mlir::Location term_loc) {
    TrivialReshaper reshaper;
    return takeAdjointOfBlockInPlace<TerminatorType>(
        rewriter, reshaper, fwd_block, term_loc);
}

// Same as takeAdjointOfBlockInPlace() except takes some pretend block_args
// like takeAdjointOfBlockInPlaceNoTerm() above.
template<typename TerminatorType>
mlir::LogicalResult takeAdjointOfBlockInPlace(
        mlir::RewriterBase &rewriter,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        mlir::Location term_loc) {
    TrivialReshaper reshaper;
    return takeAdjointOfBlockInPlace<TerminatorType>(
        rewriter, reshaper, fwd_block, block_args, term_loc);
}

// Same as takeAdjointOfBlockInPlace() above except also takes a reshaper.
template<typename TerminatorType>
mlir::LogicalResult takeAdjointOfBlockInPlace(
        mlir::RewriterBase &rewriter,
        Reshaper &reshaper,
        mlir::Block &fwd_block,
        mlir::Location term_loc) {
    llvm::SmallVector<mlir::Value> actual_block_args(
        fwd_block.getArguments());
    return takeAdjointOfBlockInPlace<TerminatorType>(
        rewriter, reshaper, fwd_block, actual_block_args, term_loc);
}

// Same as takeAdjointOfBlockInPlace() above except also takes a reshaper and
// pretend block args.
template<typename TerminatorType>
mlir::LogicalResult takeAdjointOfBlockInPlace(
        mlir::RewriterBase &rewriter,
        Reshaper &reshaper,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        mlir::Location term_loc) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    llvm::SmallVector<mlir::Value> term_args;
    if (takeAdjointOfBlockInPlaceNoTerm(rewriter, reshaper, fwd_block,
                                        block_args, term_args).failed()) {
        return mlir::failure();
    }
    rewriter.setInsertionPointToEnd(&fwd_block);
    rewriter.create<TerminatorType>(term_loc, mlir::TypeRange(), term_args);
    return mlir::success();
}

// Circuit synthesis

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
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx);

} // namespace qcirc

#endif // DIALECT_INCLUDE_QCIRC_UTILS_UTILS_H
