#ifndef DIALECT_INCLUDE_CCIRC_SYNTH_CCIRC_SYNTH_H
#define DIALECT_INCLUDE_CCIRC_SYNTH_CCIRC_SYNTH_H

#include "CCirc/IR/CCircOps.h"

// Classical synthesis routines for the CCirc dialect

namespace ccirc {

// Synthesize `wire_cond? wire_then : wire_else` where each wire is one bit.
mlir::Value synthIfElse(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::Value wire_cond,
        mlir::Value wire_then,
        mlir::Value wire_else);

// Synthesize `wire_cond? wires_then : wires_else`.
void synthMux(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::Value wire_cond,
        llvm::SmallVectorImpl<mlir::Value> &wires_then,
        llvm::SmallVectorImpl<mlir::Value> &wires_else,
        llvm::SmallVectorImpl<mlir::Value> &wires_out);

enum class BitRotateDirection {
    Left,
    Right,
};

// Synthesize classical circuitry that achieves rotr(n, k) or rotl(n, k).
// k is assumed to be big endian, and higher-order bits beyond ceil(log2(n))
// are ignored.
void synthBitRotate(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        BitRotateDirection kind,
        llvm::SmallVectorImpl<mlir::Value> &wires_n,
        llvm::SmallVectorImpl<mlir::Value> &wires_k,
        llvm::SmallVectorImpl<mlir::Value> &wires_out);

} // namespace ccirc

#endif // DIALECT_INCLUDE_CCIRC_SYNTH_CCIRC_SYNTH_H
