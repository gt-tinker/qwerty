// Must be first include. See util.hpp
#include "util.hpp"

#include <algorithm>

#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"

namespace ccirc {

// This circuit is from create_ite() in tweedledum:
// https://github.com/boschmitt/tweedledum/blob/9d3a2fab17e8531e1edc0ab927397d449b9942a4/external/mockturtle/mockturtle/networks/abstract_xag.hpp#L362-L378
mlir::Value synthIfElse(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::Value wire_cond,
        mlir::Value wire_then,
        mlir::Value wire_else) {
    // This is a 1-bit mux: `cond? then : else`. This is implemented as
    // `((then ^ else) & ~cond) ^ then`. You can verify the correctness of this
    // decomposition with a truth table, but here is my intuition for it:
    //
    // 1. `then ^ else` is 1 iff `then` and `else` differ. Note that if this is
    //    (i.e., if `then` and `else` differ), then flipping `then` will produce
    //    `else`.
    // 2. If `cond == 0` (that is, if `~cond` is true), then we want the output
    //    to be `else`.
    // 3. Thus, we should return `then` if `(then ^ else) & ~cond` is 1, else
    //    we should return `~then`. The expression
    //    `((then ^ else) & ~cond) ^ then` achieves exactly this.
    //
    // Why do it this way? ANDs (Toffolis) are dramatically more expensive than
    // XORs (CNOTs), both in terms of ancilla and T gates.

    mlir::Value then_and_else_differ = builder.create<ccirc::XorOp>(
        loc, wire_then, wire_else).getResult();

    mlir::Value else_desired = builder.create<ccirc::NotOp>(
        loc, wire_cond).getResult();

    mlir::Value should_flip_then = builder.create<ccirc::AndOp>(
        loc, else_desired, then_and_else_differ).getResult();

    mlir::Value maybe_flipped_then = builder.create<ccirc::XorOp>(
        loc, wire_then, should_flip_then).getResult();

    return maybe_flipped_then;
}

// Idea taken from tweedledum:
// https://github.com/boschmitt/tweedledum/blob/9d3a2fab17e8531e1edc0ab927397d449b9942a4/external/mockturtle/mockturtle/generators/control.hpp#L86-L110
void synthMux(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::Value wire_cond,
        llvm::SmallVectorImpl<mlir::Value> &wires_then,
        llvm::SmallVectorImpl<mlir::Value> &wires_else,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    assert(wires_then.size() == wires_else.size()
           && "Wire dimension mismatch between if/else branch");

    wires_out.clear();
    for (size_t i = 0; i < wires_then.size(); i++) {
        wires_out.push_back(
            synthIfElse(builder, loc, wire_cond, wires_then[i],
                        wires_else[i]));
    }
}

void synthBitRotate(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        BitRotateDirection direction,
        llvm::SmallVectorImpl<mlir::Value> &wires_n,
        llvm::SmallVectorImpl<mlir::Value> &wires_k,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    // Calculate the number of bits needed to specify k in theory. This is
    // ceil(log2(n))
    size_t num_bits = wires_n.size();
    assert(num_bits && "n is zero bits???");
    // Round number of bits up to a power of 2
    if (BITS_POPCOUNT(num_bits) > 1) {
        num_bits = 1ULL << BITS_NEEDED(num_bits);
    }
    // Then find the index of the most significant 1, which should be log2 of
    // the number
    size_t k_bits = BITS_NEEDED(num_bits)-1;
    assert(k_bits && "No bits needed for k? How?");

    wires_out.clear();
    wires_out.append(wires_n.begin(), wires_n.end());

    // Barrel shifter:
    // https://www.d.umn.edu/~gshute/logic/barrel-shifter.html
    for (size_t i = 0; i < std::min(k_bits, wires_k.size()); i++) {
        llvm::SmallVector<mlir::Value> unshifted(
            wires_out.begin(), wires_out.end());
        llvm::SmallVector<mlir::Value> shifted(unshifted);
        size_t middle_idx = 1ULL << i;
        if (direction == BitRotateDirection::Right) {
            // A right rotate by k is a left rotate by n_bits-k
            middle_idx = shifted.size() - middle_idx;
        }
        std::rotate(shifted.begin(), shifted.begin() + middle_idx,
                    shifted.end());
        // Assume k is big endian
        synthMux(builder, loc, wires_k[wires_k.size()-1-i], shifted, unshifted,
                 wires_out);
    }
}

} // namespace ccirc
