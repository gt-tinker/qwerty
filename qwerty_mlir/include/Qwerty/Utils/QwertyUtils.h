#ifndef DIALECT_INCLUDE_QWERTY_UTILS_UTILS_H
#define DIALECT_INCLUDE_QWERTY_UTILS_UTILS_H

#include "mlir/IR/PatternMatch.h"

#include "QCirc/Utils/QCircUtils.h"

// Miscellaneous utilities for the Qwerty dialect, currently just for
// predication.

namespace qwerty {

// Qwerty's concept of predication is more general than the typical
// definition of controls in a circuit. Thus, to synthesize the circuit for
// b & f for some basis b and function f, the body of f may need to be
// repeated multiple times in different subspaces. (There are example of this
// where f is a bit flip below.)
//
// This function unfurls a predication into traditional controls
// conjugated with bit flips and basis change gates as needed. It is not
// aware of whatever you are trying to predicate — your callback is
// responsible for applying the body of f with the desired controls — but
// this function will handle conjugating the controls for you. In the
// examples below, for instance, the gates on the upper 3 qubits will be
// added for you, but it is your responsibility to insert the CCCX gate, that
// is, the controlled version of the original operation. It is crucial for
// the callback to update the vector of controls qubits in-place with the
// results of running any gates on the controls to avoid violating qubit
// linearity, which would corrupt the IR.
//
// If there are no predicates in the basis provided, the callback will be
// called once with an empty list of control qubits.
//
// Examples
// --------
// The operation std.flip in Qwerty is lowered to a circuit with just a bit
// flip gate:
//     --X--
// Thus, you can imagine a Qwerty operation such as
//     {'110', '001'} & std.flip
// being lowered to the following circuit so that the bit flip runs in the
// subspace span(|110⟩,|001⟩)⊗ 𝓗_2 (below, "◯" means "controlled on
// zero" and "⬤" means "controlled on one"):
//     --⬤--◯--
//       |  |
//     --⬤--◯--
//       |  |
//     --◯--⬤--
//       |  |
//     --X--X--
// But zero-controls don't exist in QCirc IR, so really this would look like
// the following:
//     -----⬤-----X--⬤--X--
//          |        |
//     -----⬤-----X--⬤--X--
//          |        |
//     --X--⬤--X-----⬤-----
//          |        |
//     -----X--------X-----
// Similarly, the Qwerty syntax
//     {'mmp', 'ppm'} & std.flip
// would be lowered to the same circuit as above, just with the controls
// conjugated with Hadamards:
//     --H-----⬤-----X--⬤--X--H--
//             |        |
//     --H-----⬤-----X--⬤--X--H--
//             |        |
//     --H--X--⬤--X-----⬤-----H--
//             |        |
//     --------X--------X--------
void lowerPredBasisToControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        BasisAttr pred_basis,
        mlir::ValueRange controls_in,
        llvm::SmallVectorImpl<mlir::Value> &controls_out,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb);

// This is similar to lowerPredBasisToControls() except designed for lowering
// basis translations to circuits. The qubit indices in the basis passed
// match 1:1 with the mlir::Values (each a qubit) in the qubits vector. The
// predicates are extracted from the basis provided, the corresponding qubits
// for those predicates are extracted from the list of qubits, and then
// lowerPredBasisToControls() is invoked with the callback provided.
//
// Example
// -------
// First, observe that std >> pm can be lowered to a circuit as
//     --H--
// Then a version of that basis translation with predicates, such as
//     {'11'} + std + {'1'} >> {'11'} + pm + {'1'}
// would get compiled to
//     --⬤--
//       |
//     --⬤--
//       |
//     --H--
//       |
//     --⬤--
// When using this subroutine, you would pass a vector of 4 qubits. The
// callback would get called once with a vector of 3 qubits (the
// second-to-last qubit is excluded, since it wouldn't be a control).
void lowerPredBasisToInterleavedControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        BasisAttr pred_basis,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb);

// Same as lowerPredBasisToInterleavedControls() except it allows the caller
// to pass basis elements directly instead of a BasisAttr.
void lowerPredBasisToInterleavedControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::ArrayRef<qwerty::BasisElemAttr> pred_elems,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb);

// Predicate a basic block in-place on the basis pred_basis with predicate
// qubits in pred_qbundle. The bundle of predicate qubits is returned via
// pred_qbundle_out. (The block terminator is untouched.) This predication is
// accomplished with the Predicatable and Controllable op interfaces (in that
// order of preference).
mlir::LogicalResult predicateBlockInPlace(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Value &pred_qbundle_out);

// Same as predicateBlockInPlace() except that the location loc will be used
// when inserting inferred SWAPs (see Section 5.3 of the CGO paper).
mlir::LogicalResult predicateBlockInPlace(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Location loc,
        mlir::Value &pred_qbundle_out);

// Same as predicateBlockInPlace() above except with two differences
// invaluable for inlining: (1) the block_args provided are imagined to be
// the block arguments (necessary because the MLIR inliner only calls our
// code after replacing all uses of block args with caller args); and (2)
// predication begins only at the op start_at (necessary because our inlining
// code does some initial unpacking and repacking that confuses the SWAP
// analysis).
mlir::LogicalResult predicateBlockInPlace(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Operation *start_at,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        mlir::Location loc,
        mlir::Value &pred_qbundle_out);

// Same as predicateBlockInPlace() above except that it also fixes the
// terminator by replacing it with a new terminator of type TerminatorType.
// The new terminator's operand is the predicate qbundle merged with the
// original result qbundle.
template<typename TerminatorType>
mlir::LogicalResult predicateBlockInPlaceFixTerm(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Operation *start_at,
        llvm::SmallVectorImpl<mlir::Value> &block_args) {
    mlir::Operation *term = block.getTerminator();
    assert(llvm::isa<TerminatorType>(term));

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    if (mlir::failed(predicateBlockInPlace(
            pred_basis, pred_qbundle, rewriter, block, start_at,
            block_args, term->getLoc(), pred_qbundle))) {
        return mlir::failure();
    }

    assert(term->getNumOperands() == 1
           && llvm::isa<QBundleType>(term->getOperand(0).getType()));
    mlir::Value ret_qbundle = term->getOperand(0);
    rewriter.setInsertionPoint(term);

    mlir::ValueRange pred_unpacked =
        rewriter.create<QBundleUnpackOp>(
            term->getLoc(), pred_qbundle).getQubits();
    mlir::ValueRange ret_unpacked =
        rewriter.create<QBundleUnpackOp>(
            term->getLoc(), ret_qbundle).getQubits();
    llvm::SmallVector<mlir::Value> all_qubits(pred_unpacked);
    all_qubits.append(ret_unpacked.begin(), ret_unpacked.end());
    mlir::Value repacked =
        rewriter.create<QBundlePackOp>(
            term->getLoc(), all_qubits).getQbundle();
    rewriter.replaceOpWithNewOp<TerminatorType>(term, repacked);
    return mlir::success();
}

} // namespace qwerty

#endif // DIALECT_INCLUDE_QWERTY_UTILS_UTILS_H
