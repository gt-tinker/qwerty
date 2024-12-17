// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "PassDetail.h"

// Replace gates that the resource estimator considers expensive with cheaper
// versions. Right now, this is just replacing P(θ) with Rz(θ).

namespace {

class ReplaceGate1Q1POpPattern : public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate, mlir::PatternRewriter &rewriter) const override {
        if (!gate.getControls().empty()
                || gate.getGate() != qcirc::Gate1Q1P::P) {
            return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rz, gate.getParam(), mlir::ValueRange(), gate.getQubit());
        return mlir::success();
    }
};

struct ReplaceNonQasmGatesPass : public qcirc::ReplaceNonQasmGatesBase<ReplaceNonQasmGatesPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceGate1Q1POpPattern
        >(&getContext());

        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createReplaceNonQasmGatesPass() {
    return std::make_unique<ReplaceNonQasmGatesPass>();
}
