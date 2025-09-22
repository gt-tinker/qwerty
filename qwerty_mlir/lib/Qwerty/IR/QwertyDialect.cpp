//===- QwertyDialect.cpp - Qwerty dialect ---------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/InliningUtils.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/Utils/QCircUtils.h"
#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyAttributes.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/IR/QwertyTypes.h"
#include "Qwerty/Utils/QwertyUtils.h"

using namespace qwerty;

#include "Qwerty/IR/QwertyOpsDialect.cpp.inc"
#include "Qwerty/IR/QwertyOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Qwerty dialect.
//===----------------------------------------------------------------------===//

namespace {

// The MLIR inliner calls into this, and we use it to predicate or take the
// adjoint for predicated/adjointed calls
struct QwertyInlinerInterface : public mlir::DialectInlinerInterface {
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    // All Qwerty callables can be inlined
    bool isLegalToInline(mlir::Operation *, mlir::Operation *, bool) const final {
        return true;
    }

    // All Qwerty dialect ops can be inlined.
    bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::IRMapping &) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                         bool wouldBeCloned, mlir::IRMapping &valueMapping) const final {
        return true;
    }

    mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder,
                                               mlir::Value input,
                                               mlir::Type resultType,
                                               mlir::Location conversionLoc) const final {
        if (!llvm::isa<qwerty::QBundleType>(resultType)
                || !llvm::isa<qwerty::QBundleType>(input.getType())) {
            return nullptr;
        }

        return builder.create<mlir::UnrealizedConversionCastOp>(
            conversionLoc, resultType, input);
    }

    void handleTerminator(mlir::Operation *op, mlir::ValueRange valuesToReplace) const final {
        qwerty::ReturnOp ret = llvm::cast<qwerty::ReturnOp>(op);
        assert(valuesToReplace.size() == ret.getOperands().size()
               && "Return op has wrong number of operands");
        for (size_t i = 0; i < valuesToReplace.size(); i++) {
            valuesToReplace[i].replaceAllUsesWith(ret.getOperands()[i]);
        }
    }

    void processInlinedCallBlocks(
            mlir::Operation *call_op,
            llvm::iterator_range<mlir::Region::iterator> inlinedBlocks) const final {
        if (qwerty::CallOp call = llvm::dyn_cast<qwerty::CallOp>(call_op)) {
            assert(llvm::range_size(inlinedBlocks) == 1
                   && "Inlined region should contain at least 1 block");
            mlir::Block &block = *inlinedBlocks.begin();
            mlir::IRRewriter rewriter(call.getContext());

            if (call.getPred()) {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                mlir::ValueRange operands = call.getCapturesAndOperands();
                assert(!operands.empty());
                mlir::Value qbundle_arg = operands[operands.size()-1];
                assert(llvm::isa<QBundleType>(qbundle_arg.getType()));

                mlir::UnrealizedConversionCastOp hacky_cast_op;
                for (mlir::OpOperand &use : qbundle_arg.getUses()) {
                    mlir::UnrealizedConversionCastOp candidate;
                    mlir::Value cast_result;
                    if ((candidate = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(use.getOwner()))
                            && candidate.getInputs().size() == 1
                            && candidate.getOutputs().size() == 1
                            && candidate->getBlock() == call->getBlock()
                            // By linearity, there must be at least one use.
                            // (We check this so a stray cast doesn't vacuously
                            //  trip the isUsedOutsideOfBlock() check below.)
                            && !(cast_result = candidate.getOutputs()[0]).use_empty()
                            && !cast_result.isUsedOutsideOfBlock(&block)) {
                        hacky_cast_op = candidate;
                        break;
                    }
                }
                assert(hacky_cast_op
                       && "No cast generated for predicated function");

                // Let's be conservative here and not modify any IR outside the
                // block we were passed
                rewriter.setInsertionPointToStart(&block);

                // Predicate bundle + argument bundle combined
                mlir::Value combined_qbundle = hacky_cast_op.getInputs()[0];
                assert(combined_qbundle == qbundle_arg);
                size_t pred_dim = call.getPredAttr().getDim();
                mlir::ValueRange unpacked =
                    rewriter.create<qwerty::QBundleUnpackOp>(
                        call.getLoc(), combined_qbundle).getQubits();
                mlir::Value pred_qbundle = rewriter.create<qwerty::QBundlePackOp>(
                        call.getLoc(),
                        llvm::iterator_range(unpacked.begin(), unpacked.begin()+pred_dim)
                    ).getQbundle();
                qwerty::QBundlePackOp arg_qbundle_op =
                    rewriter.create<qwerty::QBundlePackOp>(
                        call.getLoc(),
                        llvm::iterator_range(unpacked.begin()+pred_dim,
                                             unpacked.end()));
                mlir::Value arg_qbundle = arg_qbundle_op.getQbundle();

                mlir::Value hacky_cast_result = hacky_cast_op.getOutputs()[0];
                // We just verified above that all uses are inside the block we
                // were passed, so this should be safe
                hacky_cast_result.replaceAllUsesWith(arg_qbundle);
                // At this point, the cast has no uses, so we might as well
                // remove it to avoid violating linearity
                rewriter.eraseOp(hacky_cast_op);

                // It is tempting to take the adjoint right now, but the
                // problem is that the qbundle containing the predicate qubits
                // that we just created violates linearity. We could trivially
                // thread it through and then take the adjoint, but that would
                // require careful trickery later to recover what the
                // predication qbundle is. So just predicate now and then
                // adjoint after — in theory, the operations of adding
                // predication and taking the adjoint commute anyway.

                // This malformed state of the IR also affects the analysis
                // that predication will do. So we need to trick the analysis
                // into thinking all the crazy hacking we just did didn't
                // happen. First, tell it the last block argument is actually
                // the repacked argument qubits:
                llvm::SmallVector<mlir::Value> pretend_block_args(
                    call.getCapturesAndOperands());
                assert(!pretend_block_args.empty());
                pretend_block_args[pretend_block_args.size()-1] = arg_qbundle;
                // Next, tell the analysis to start looking at the op after the
                // arg repacking
                mlir::Operation *start_at = arg_qbundle_op->getNextNode();
                [[maybe_unused]] auto res =
                    qwerty::predicateBlockInPlaceFixTerm<qwerty::ReturnOp>(
                        call.getPredAttr(), pred_qbundle, rewriter, block,
                        start_at, pretend_block_args);
                assert(mlir::succeeded(res) && "Predicating must succeed");

                // At this point, if this call was predicated, the IR is
                // finally in a valid form and we can take the adjoint.
            }

            if (call.getAdj()) {
                llvm::SmallVector<mlir::Value> pretend_block_args(
                    call.getCapturesAndOperands());
                [[maybe_unused]] auto res =
                    qcirc::takeAdjointOfBlockInPlace<qwerty::ReturnOp>(
                        rewriter, block, pretend_block_args, call.getLoc());
                assert(mlir::succeeded(res) && "Taking adjoint must succeed");
            }
        }
    }
};
} // namespace

void QwertyDialect::initialize() {
    registerAttributes();
    registerTypes();

    addOperations<
#define GET_OP_LIST
#include "Qwerty/IR/QwertyOps.cpp.inc"
    >();

    addInterfaces<QwertyInlinerInterface>();
}
