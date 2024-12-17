#include <queue>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tweedledum/Utils/Numbers.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "PassDetail.h"

// This pass replaces every qcirc.adj op with its contents (a single basic
// block), except taking the adjoint of the block. This pass (and the
// qcirc.adj op itself) basically only exist to test
// takeAdjointOfBlockInPlace() using FileCheck tests.

namespace {

class InlineAdjointPattern : public mlir::OpRewritePattern<qcirc::AdjointOp> {
    using mlir::OpRewritePattern<qcirc::AdjointOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::AdjointOp adj, mlir::PatternRewriter &rewriter) const override {
        mlir::Block &block = adj.getRegion().front();
        if (qcirc::takeAdjointOfBlockInPlace<qcirc::YieldOp>(
                rewriter, block, adj.getLoc()).failed()) {
            return mlir::failure();
        }
        qcirc::YieldOp yield =
            llvm::cast<qcirc::YieldOp>(block.getTerminator());
        rewriter.inlineBlockBefore(&block, adj, adj.getInputs());
        rewriter.replaceOp(adj, yield.getQubits());
        rewriter.eraseOp(yield);
        return mlir::success();
    }
};

struct InlineAdjPass : public qcirc::InlineAdjBase<InlineAdjPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<InlineAdjointPattern>(&getContext());

        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createInlineAdjPass() {
    return std::make_unique<InlineAdjPass>();
}
