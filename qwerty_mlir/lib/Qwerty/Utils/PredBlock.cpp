#include "QCirc/IR/QCircOps.h"
#include "QCirc/IR/QCircInterfaces.h"
#include "QCirc/Utils/QCircUtils.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/IR/QwertyInterfaces.h"
#include "Qwerty/Utils/QwertyUtils.h"
#include "Qwerty/Analysis/QubitIndexAnalysis.h"

namespace {

bool isIdentity(const qwerty::QubitIndexVec::Indices &inds) {
    for (auto [i, pi_i] : llvm::enumerate(inds)) {
        if (i != pi_i) {
            return false;
        }
    }
    return true;
}

struct PredElem {
    size_t offset;
    qwerty::BasisVectorListAttr veclist;

    PredElem(size_t offset, qwerty::BasisVectorListAttr veclist)
           : offset(offset), veclist(veclist) {}
};

void findPredElems(llvm::ArrayRef<qwerty::BasisElemAttr> elems,
                  llvm::SmallVectorImpl<PredElem> &pred_elems_out) {
    size_t offset = 0;
    for (qwerty::BasisElemAttr elem : elems) {
        if (elem.isPredicate()) {
            qwerty::BasisVectorListAttr vl = elem.getVeclist();
            assert(vl);
            pred_elems_out.emplace_back(offset, vl);
        }
        offset += elem.getDim();
    }
}

void findPredElems(qwerty::BasisAttr pred_basis,
                  llvm::SmallVectorImpl<PredElem> &pred_elems_out) {
    findPredElems(pred_basis.getElems(), pred_elems_out);
}

// Returns true when done
bool nextInCartesianProduct(llvm::SmallVectorImpl<PredElem> &pred_elems,
                            llvm::SmallVectorImpl<size_t> &i) {
    size_t n_elems = pred_elems.size();

    size_t j;
    for (j = 0; j < n_elems && i[j] == pred_elems[j].veclist.getVectors().size()-1; j++);

    bool done;
    if ((done = j == n_elems)) {
        // Done!
    } else {
        for (size_t k = 0; k < j; k++) {
            i[k] = 0;
        }
        i[j]++;
    }

    return done;
};

void getStationaryOperands(
        qcirc::AdjointableOpInterface adj,
        llvm::SmallVectorImpl<mlir::Value> &stationary_operands_out) {
    stationary_operands_out.clear();
    for (size_t i = 0; i < adj->getNumOperands(); i++) {
        if (adj.isStationaryOperand(i)) {
            stationary_operands_out.push_back(adj->getOperand(i));
        }
    }
}

void standardizeAndFlip(mlir::OpBuilder &builder,
                        mlir::Location loc,
                        llvm::SmallVectorImpl<mlir::Value> &controls,
                        llvm::SmallVectorImpl<size_t> &i,
                        llvm::SmallVectorImpl<PredElem> &pred_elems) {
    for (size_t j = 0; j < i.size(); j++) {
        size_t offset = pred_elems[j].offset;
        qwerty::BasisVectorAttr vec = pred_elems[j].veclist.getVectors()[i[j]];

        for (size_t k = 0; k < vec.getDim(); k++) {
            mlir::Value ctrl = controls[offset + k];

            if (vec.getPrimBasis() == qwerty::PrimitiveBasis::X) {
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::H, mlir::ValueRange(), ctrl
                    ).getResult();
            } else if (vec.getPrimBasis() == qwerty::PrimitiveBasis::Y) {
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(), ctrl
                    ).getResult();
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::H, mlir::ValueRange(), ctrl
                    ).getResult();
            } else {
                assert(vec.getPrimBasis() == qwerty::PrimitiveBasis::Z
                       && "Missing prim_basis in PredBlock");
            }

            bool bit = vec.getEigenbits()[vec.getDim()-1-k];
            if (!bit) {
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, mlir::ValueRange(), ctrl
                    ).getResult();
            }

            controls[offset + k] = ctrl;
        }
    }
}

// Adjoint of flipAndStandardize()
void unflipAndDestandardize(mlir::OpBuilder &builder,
                            mlir::Location loc,
                            llvm::SmallVectorImpl<mlir::Value> &controls,
                            llvm::SmallVectorImpl<size_t> &i,
                            llvm::SmallVectorImpl<PredElem> &pred_elems) {
    for (size_t j = 0; j < i.size(); j++) {
        size_t offset = pred_elems[j].offset;
        qwerty::BasisVectorAttr vec = pred_elems[j].veclist.getVectors()[i[j]];

        for (size_t k = 0; k < vec.getDim(); k++) {
            mlir::Value ctrl = controls[offset + k];

            bool bit = vec.getEigenbits()[vec.getDim()-1-k];
            if (!bit) {
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, mlir::ValueRange(), ctrl
                    ).getResult();
            }

            if (vec.getPrimBasis() == qwerty::PrimitiveBasis::X) {
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::H, mlir::ValueRange(), ctrl
                    ).getResult();
            } else if (vec.getPrimBasis() == qwerty::PrimitiveBasis::Y) {
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::H, mlir::ValueRange(), ctrl
                    ).getResult();
                ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::S, mlir::ValueRange(), ctrl
                    ).getResult();
            } else {
                assert(vec.getPrimBasis() == qwerty::PrimitiveBasis::Z
                       && "Missing prim_basis in PredBlock");
            }

            controls[offset + k] = ctrl;
        }
    }
}

void borrowActualControls(
        llvm::SmallVectorImpl<PredElem> &pred_elems,
        llvm::SmallVectorImpl<mlir::Value> &controls,
        llvm::SmallVectorImpl<mlir::Value> &actual_controls_out) {
    actual_controls_out.clear();
    for (PredElem &pred_elem : pred_elems) {
        actual_controls_out.append(
            controls.begin() + pred_elem.offset,
            controls.begin() + pred_elem.offset + pred_elem.veclist.getDim());
    }
}

void restoreActualControls(
        llvm::SmallVectorImpl<PredElem> &pred_elems,
        llvm::SmallVectorImpl<mlir::Value> &controls_out,
        llvm::SmallVectorImpl<mlir::Value> &actual_controls) {
    size_t actual_idx = 0;
    for (PredElem &pred_elem : pred_elems) {
        size_t vl_dim = pred_elem.veclist.getDim();
        for (size_t i = 0; i < vl_dim; i++) {
            controls_out[pred_elem.offset + i] = actual_controls[actual_idx++];
        }
    }
}

void lowerPredBasisToControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<PredElem> &pred_elems,
        mlir::ValueRange controls_in,
        llvm::SmallVectorImpl<mlir::Value> &controls_out,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb) {
    if (pred_elems.empty()) {
        // Avoid tricky situations below and just call with an empty list of
        // control qubits
        llvm::SmallVector<mlir::Value> empty{};
        cb(empty);
        controls_out.clear();
        controls_out.append(controls_in.begin(), controls_in.end());
    } else {
        llvm::SmallVector<mlir::Value> controls(controls_in);
        llvm::SmallVector<size_t> i(pred_elems.size(), 0);

        do {
            standardizeAndFlip(builder, loc, controls, i, pred_elems);

            llvm::SmallVector<mlir::Value> actual_controls;
            borrowActualControls(pred_elems, controls, actual_controls);
            cb(actual_controls);
            restoreActualControls(pred_elems, controls, actual_controls);

            unflipAndDestandardize(builder, loc, controls, i, pred_elems);
        } while (!nextInCartesianProduct(pred_elems, i));

        controls_out = controls;
    }
}

size_t getInputDim(llvm::SmallVectorImpl<mlir::Value> &block_args) {
    assert(!block_args.empty() && "reversible block with no arguments??");
    mlir::Value last_block_arg = block_args[block_args.size()-1];
    return llvm::cast<qwerty::QBundleType>(last_block_arg.getType()).getDim();
}

using Swap = std::pair<size_t, size_t>;

void inferSwaps(mlir::RewriterBase &rewriter,
                mlir::Operation *start_at,
                llvm::SmallVectorImpl<mlir::Value> &block_args,
                size_t input_dim,
                mlir::Location loc,
                mlir::Block &block,
                llvm::SmallVectorImpl<Swap> &swaps_out) {
    qwerty::QubitIndexAnalysis analysis = qwerty::runQubitIndexAnalysis(
        block, start_at, block_args);
    mlir::Operation *term = block.getTerminator();
    assert(term->getNumOperands() == 1 && "should return 1 qbundle");
    mlir::Value ret_qbundle = term->getOperand(0);
    assert(ret_qbundle.hasOneUse() && "return qbundle violates linearity");

    assert(analysis.count(ret_qbundle)
           && "Analysis missing for returned qbundle");
    const qwerty::QubitIndexVec &inds = analysis.at(ret_qbundle);
    assert(!inds.isBottom() && "Analysis too imprecise to proceed");
    assert(inds.indices.size() == input_dim
           && "Wrong number of qubits returned. "
              "How did typechecking miss this?");
#ifndef NDEBUG
    for (size_t ind : inds.indices) {
        assert(ind < input_dim && "Permutation outside of range");
    }
#endif

    swaps_out.clear();
    if (isIdentity(inds.indices)) {
        // Nothing to do, we are in good shape!
    } else {
        llvm::SmallVector<size_t> perm = inds.indices;

        // https://stackoverflow.com/a/76399616/321301
        for (size_t i = 0; i < perm.size(); i++) {
            while (perm[i] != i) {
                size_t j = perm[i];
                swaps_out.emplace_back(i, j);
                std::swap(perm[i], perm[j]);
            }
        }

        std::reverse(swaps_out.begin(), swaps_out.end());
    }
}

mlir::Value rewriteWithSwaps(mlir::RewriterBase &rewriter,
                             llvm::SmallVectorImpl<Swap> &swaps,
                             // Only relevant qubits to predicate on. Useful when there are basis
                             // elements that fully span
                             llvm::SmallVectorImpl<PredElem> &pred_elems,
                             mlir::Value pred_qbundle,
                             size_t input_dim,
                             mlir::Operation *term,
                             mlir::Location loc) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Value ret_qbundle = term->getOperand(0);
    rewriter.setInsertionPoint(term);
    qwerty::QBundleUnpackOp unpack =
        rewriter.create<qwerty::QBundleUnpackOp>(loc, ret_qbundle);

    qwerty::QBundleUnpackOp pred_unpack =
        rewriter.create<qwerty::QBundleUnpackOp>(loc, pred_qbundle);

    mlir::ValueRange unpacked = unpack.getQubits();
    llvm::SmallVector<mlir::Value> qubits(unpacked);
    llvm::SmallVector<mlir::Value> pred_qubits(pred_unpack.getQubits());

    for (auto [i, j] : swaps) {
        qcirc::Gate2QOp swap =
            rewriter.create<qcirc::Gate2QOp>(
                loc, qcirc::Gate2Q::Swap,
                mlir::ValueRange(), qubits[i], qubits[j]);
        qubits[i] = swap.getLeftResult();
        qubits[j] = swap.getRightResult();

        lowerPredBasisToControls(
            rewriter,
            loc,
            pred_elems,
            pred_qubits,
            pred_qubits,
            [&](llvm::SmallVectorImpl<mlir::Value> &actual_controls) {
                qcirc::Gate2QOp pred_swap =
                    rewriter.create<qcirc::Gate2QOp>(
                        loc, qcirc::Gate2Q::Swap,
                        actual_controls, qubits[i], qubits[j]);
                qubits[i] = pred_swap.getLeftResult();
                qubits[j] = pred_swap.getRightResult();
                actual_controls.clear();
                actual_controls.append(pred_swap.getControlResults().begin(), pred_swap.getControlResults().end());
            }
        );
    }

    mlir::Value repacked =
        rewriter.create<qwerty::QBundlePackOp>(
            loc, qubits).getQbundle();

    mlir::Value pred_repacked =
        rewriter.create<qwerty::QBundlePackOp>(
            loc, pred_qubits).getQbundle();

    rewriter.replaceAllUsesExcept(ret_qbundle, repacked, unpack);

    return pred_repacked;
}

} // namespace

namespace qwerty {

void lowerPredBasisToControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        qwerty::BasisAttr pred_basis,
        mlir::ValueRange controls_in,
        llvm::SmallVectorImpl<mlir::Value> &controls_out,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb) {
    llvm::SmallVector<PredElem> pred_elems;
    findPredElems(pred_basis, pred_elems);

    lowerPredBasisToControls(builder, loc, pred_elems,
                             controls_in, controls_out, cb);
}

void lowerPredBasisToInterleavedControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::ArrayRef<qwerty::BasisElemAttr> elems,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb) {
#ifndef NDEBUG
    size_t total_dim = 0;
    for (qwerty::BasisElemAttr elem : elems) {
        total_dim += elem.getDim();
    }
    assert(total_dim == qubits.size()
           && "Invariant of interleaved controls violated");
#endif

    llvm::SmallVector<PredElem> pred_elems;
    findPredElems(elems, pred_elems);

    llvm::SmallVector<mlir::Value> controls;
    for (PredElem &elem : pred_elems) {
        controls.append(qubits.begin() + elem.offset,
                        qubits.begin() + elem.offset + elem.veclist.getDim());
    }

    lowerPredBasisToControls(builder, loc, pred_elems, controls, controls, cb);

    size_t qubit_idx = 0;
    for (PredElem &elem : pred_elems) {
        for (size_t i = 0; i < elem.veclist.getDim(); i++) {
            qubits[elem.offset + i] = controls[qubit_idx];
            qubit_idx++;
        }
    }
}

void lowerPredBasisToInterleavedControls(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        BasisAttr pred_basis,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        std::function<void(llvm::SmallVectorImpl<mlir::Value>&)> cb) {
    return lowerPredBasisToInterleavedControls(
        builder, loc, pred_basis? pred_basis.getElems()
                                // empty if pred_basis is NULL
                                : llvm::ArrayRef<qwerty::BasisElemAttr>(),
        qubits, cb);
}

mlir::LogicalResult predicateBlockInPlace(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Location loc,
        mlir::Value &pred_qbundle_out) {
    llvm::SmallVector<mlir::Value> actual_block_args(block.getArguments());
    mlir::Operation *first_op = &block.front();
    return predicateBlockInPlace(pred_basis, pred_qbundle, rewriter,
                                 block, first_op, actual_block_args, loc,
                                 pred_qbundle_out);
}

mlir::LogicalResult predicateBlockInPlace(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Value &pred_qbundle_out) {
    return predicateBlockInPlace(pred_basis, pred_qbundle, rewriter,
                                 block, rewriter.getUnknownLoc(),
                                 pred_qbundle_out);
}

mlir::LogicalResult predicateBlockInPlace(
        BasisAttr pred_basis,
        mlir::Value pred_qbundle,
        mlir::RewriterBase &rewriter,
        mlir::Block &block,
        mlir::Operation *start_at,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        mlir::Location loc,
        mlir::Value &pred_qbundle_out) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    size_t input_dim = getInputDim(block_args);

    llvm::SmallVector<Swap> swaps;
    inferSwaps(rewriter, start_at, block_args, input_dim, loc, block, swaps);

    llvm::SmallVector<PredElem> pred_elems;
    findPredElems(pred_basis, pred_elems);

    mlir::Operation *op = start_at;

    while (op) {
        // Before we delete this op, get a pointer to the next one
        mlir::Operation *next = op->getNextNode();

        if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
            // Predicating a terminator doesn't make much sense. Let him cook
            // peacefully
        } else if (op->hasTrait<qcirc::IsStationaryOpTrait>()) {
            // Leave this guy alone. He claims no qubits are flowing through
            // him.
        } else if (PredicatableOpInterface predicatable =
                llvm::dyn_cast<PredicatableOpInterface>(op)) {
            if (!predicatable.isPredicatable()) {
                return op->emitOpError("not predicatable");
            }

            if (pred_elems.empty()) {
                // If there are no predicates, don't even bother messing with
                // these guys
            } else {
                llvm::SmallVector<mlir::Value> new_outputs;
                rewriter.setInsertionPoint(op);
                predicatable.buildPredicated(
                    rewriter, pred_basis, pred_qbundle, pred_qbundle,
                    op->getOperands(), new_outputs);
                assert(new_outputs.size() == op->getNumResults());
                if (mlir::ValueRange(new_outputs)
                        != mlir::ValueRange(op->getResults())) {
                    rewriter.replaceOp(op, new_outputs);
                }
            }
        } else if (qcirc::ControllableOpInterface controllable =
                llvm::dyn_cast<qcirc::ControllableOpInterface>(op)) {
            if (!controllable.isControllable()) {
                return op->emitOpError("not controllable");
            }

            mlir::Location loc = op->getLoc();
            rewriter.setInsertionPoint(op);
            mlir::ValueRange unpacked =
                rewriter.create<QBundleUnpackOp>(
                    loc, pred_qbundle).getQubits();

            llvm::SmallVector<mlir::Value> stationary_operands;
            getStationaryOperands(
                llvm::cast<qcirc::AdjointableOpInterface>(op),
                stationary_operands);
            // As in AdjointBlock, we operate with the assumption that the
            // stationary operands are the first operands. We'll be taking
            // the outputs and feeding them as inputs to the next round, so
            // bootstrap this for the first round by pretending the current
            // nonstationary operands are from an imaginary previous round.
            llvm::SmallVector<mlir::Value> outputs(
                op->operand_begin() + stationary_operands.size(),
                op->operand_end());

            bool no_controls = false;
            llvm::SmallVector<mlir::Value> controls;
            lowerPredBasisToControls(
                rewriter, loc, pred_elems, unpacked, controls,
                [&](llvm::SmallVectorImpl<mlir::Value> &actual_controls) {
                    if (actual_controls.empty()) {
                        // If there are no predicates, then there is no need to
                        // even add controls. Just keep chugging along and
                        // don't bother replacing `op' (see below, where we use
                        // this variable)
                        no_controls = true;
                    } else {
                        llvm::SmallVector<mlir::Value> inputs(
                            stationary_operands);
                        inputs.append(outputs);
                        controllable.buildControlled(
                            rewriter, actual_controls, actual_controls,
                            inputs, outputs);
                    }
                });

            pred_qbundle =
                rewriter.create<QBundlePackOp>(loc, controls).getQbundle();

            // This is the last thing we do because op is the insertion point.
            // Inserting anything after this will cause a segfault
            if (!no_controls) {
                rewriter.replaceOp(op, outputs);
            }
        } else {
            return op->emitOpError("not stationary, predicatable, or "
                                   "controllable");
        }

        op = next;
    }

    pred_qbundle_out = rewriteWithSwaps(
        rewriter, swaps, pred_elems, pred_qbundle, input_dim,
        block.getTerminator(), loc);

    return mlir::success();
}

} // namespace qwerty
