#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Utils/QwertyUtils.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "PassDetail.h"

// This is a pass that replaces every qwerty.pred op with its body (a single
// basic block), except with every instruction predicated. This pass (and the
// qwerty.pred op itself) basically only exist to test
// predicateBlockInPlace() using FileCheck tests.

namespace {

class InlinePredPattern : public mlir::OpRewritePattern<qwerty::PredOp> {
    using mlir::OpRewritePattern<qwerty::PredOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::PredOp pred,
                                        mlir::PatternRewriter &rewriter) const final {
        qwerty::BasisAttr basis = pred.getBasis();
        mlir::Region &region = pred.getRegion();
        assert(region.hasOneBlock());
        mlir::Block &block = region.front();

        mlir::Value pred_out;
        if (mlir::failed(qwerty::predicateBlockInPlace(
                basis, pred.getPredBundleIn(), rewriter, block,
                pred_out))) {
            return mlir::failure();
        }

        qwerty::YieldOp yield =
            llvm::cast<qwerty::YieldOp>(block.getTerminator());
        mlir::Value yielded_val = yield.getQbundle();

        rewriter.inlineBlockBefore(&block, pred, pred.getRegionArg());
        rewriter.eraseOp(yield);
        rewriter.replaceOp(pred, {pred_out, yielded_val});
        return mlir::success();
    }
};

struct InlinePredPass : public qwerty::InlinePredBase<InlinePredPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<InlinePredPattern>(&getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qwerty::createInlinePredPass() {
    return std::make_unique<InlinePredPass>();
}

