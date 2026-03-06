#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"

namespace ccirc {

void synthAdd(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_b,
        llvm::SmallVectorImpl<mlir::Value> &wires_sum) {
    // TODO: implement me
    assert(0 && "Add not implemented");
}

void synthModMul(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt x,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_y,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    // TODO: implement me
    assert(0 && "Mod mul not implemented");
}

} // namespace ccirc
