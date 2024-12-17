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

// This pass decomposes multi-controlled gates using Selinger's trick
// (https://doi.org/10.1103/PhysRevA.87.042302). When this pass finishes,
// each gate will have at most 2 controls.

namespace {

// This is due to Selinger (Equations 10 and 11): https://doi.org/10.1103/PhysRevA.87.042302
// x and y are controls, and z is the target
std::tuple<mlir::Value, mlir::Value, mlir::Value> ccxphase(
        mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value x, mlir::Value y, mlir::Value z) {
    qcirc::Gate1QOp h1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::H, mlir::ValueRange(), z);
    z = h1.getResult();

    qcirc::Gate1QOp cx1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, z, y);
    z = cx1.getControlResults()[0];
    y = cx1.getResult();

    qcirc::Gate1QOp cx2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, y, x);
    y = cx2.getControlResults()[0];
    x = cx2.getResult();

    qcirc::Gate1QOp t1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::T, mlir::ValueRange(), x);
    x = t1.getResult();
    qcirc::Gate1QOp tdg1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), y);
    y = tdg1.getResult();
    qcirc::Gate1QOp t2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::T, mlir::ValueRange(), z);
    z = t2.getResult();

    qcirc::Gate1QOp cx3 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, z, y);
    z = cx3.getControlResults()[0];
    y = cx3.getResult();

    qcirc::Gate1QOp cx4 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, y, x);
    y = cx4.getControlResults()[0];
    x = cx4.getResult();

    qcirc::Gate1QOp tdg2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), x);
    x = tdg2.getResult();

    qcirc::Gate1QOp cx5 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, z, x);
    z = cx5.getControlResults()[0];
    x = cx5.getResult();

    qcirc::Gate1QOp h2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::H, mlir::ValueRange(), z);
    z = h2.getResult();

    return {x, y, z};
}

// This is the adjoint of ccxphase() above
std::tuple<mlir::Value, mlir::Value, mlir::Value> ccxphase_adj(
        mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value x, mlir::Value y, mlir::Value z) {
    qcirc::Gate1QOp h2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::H, mlir::ValueRange(), z);
    z = h2.getResult();

    qcirc::Gate1QOp cx5 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, z, x);
    z = cx5.getControlResults()[0];
    x = cx5.getResult();

    qcirc::Gate1QOp t2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::T, mlir::ValueRange(), x);
    x = t2.getResult();

    qcirc::Gate1QOp cx4 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, y, x);
    y = cx4.getControlResults()[0];
    x = cx4.getResult();

    qcirc::Gate1QOp cx3 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, z, y);
    z = cx3.getControlResults()[0];
    y = cx3.getResult();

    qcirc::Gate1QOp t1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::T, mlir::ValueRange(), y);
    y = t1.getResult();
    qcirc::Gate1QOp tdg2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), z);
    z = tdg2.getResult();
    qcirc::Gate1QOp tdg1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), x);
    x = tdg1.getResult();

    qcirc::Gate1QOp cx2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, y, x);
    y = cx2.getControlResults()[0];
    x = cx2.getResult();

    qcirc::Gate1QOp cx1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, z, y);
    z = cx1.getControlResults()[0];
    y = cx1.getResult();

    qcirc::Gate1QOp h1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::H, mlir::ValueRange(), z);
    z = h1.getResult();

    return {x, y, z};
}

class ReplaceManyControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().size() < 3) {
            // We'll let someone downstream deal with this controlled-U (or
            // quasi-Toffoli) gate
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();
        llvm::SmallVector<mlir::Value> controls(gate.getControls());

        size_t n_ancilla = controls.size()-2;
        llvm::SmallVector<mlir::Value> ancillas;
        for (size_t i = 0; i < n_ancilla; i++) {
            ancillas.push_back(rewriter.create<qcirc::QallocOp>(loc).getResult());
        }

        std::tie(controls[0], controls[1], ancillas[0]) =
            ccxphase(rewriter, loc, controls[0], controls[1], ancillas[0]);

        for (size_t i = 1; i < ancillas.size(); i++) {
            std::tie(ancillas[i-1], controls[i+1], ancillas[i]) =
                ccxphase(rewriter, loc, ancillas[i-1], controls[i+1], ancillas[i]);
        }

        llvm::SmallVector<mlir::Value> ccu_controls{ancillas[ancillas.size()-1],
                                                    controls[controls.size()-1]};
        qcirc::Gate1QOp ccu =
            rewriter.create<qcirc::Gate1QOp>(loc,
                gate.getGate(), ccu_controls, gate.getQubit());
        ancillas[ancillas.size()-1] = ccu.getControlResults()[0];
        controls[controls.size()-1] = ccu.getControlResults()[1];
        mlir::Value ccu_tgt = ccu.getResult();

        for (size_t ii = 1; ii < ancillas.size(); ii++) {
            size_t i = ancillas.size()-1-(ii-1);
            std::tie(ancillas[i-1], controls[i+1], ancillas[i]) =
                ccxphase_adj(rewriter, loc, ancillas[i-1], controls[i+1], ancillas[i]);
        }

        std::tie(controls[0], controls[1], ancillas[0]) =
            ccxphase_adj(rewriter, loc, controls[0], controls[1], ancillas[0]);

        for (mlir::Value ancilla : ancillas) {
            rewriter.create<qcirc::QfreeZeroOp>(loc, ancilla);
        }

        llvm::SmallVector<mlir::Value> replaceWith(controls);
        replaceWith.push_back(ccu_tgt);
        rewriter.replaceOp(gate, replaceWith);

        return mlir::success();
    }
};

// Tweedledum already uses Selinger's trick when possible, but they indicate
// this with Rx(pi) and Rx(-pi) gates. We should replace those with their
// decomposition to avoid downstream compilers/transpilers interpreting it
// foolishly.
class ReplaceTweedledumCCXPhasePattern : public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate, mlir::PatternRewriter &rewriter) const override {
        mlir::FloatAttr angle_attr;
        if (gate.getControls().size() != 2
                || gate.getGate() != qcirc::Gate1Q1P::Rx
                || !mlir::matchPattern(gate.getParam(), qcirc::m_CalcConstant(&angle_attr))) {
            return mlir::failure();
        }
        bool is_adj;
        double angle = angle_attr.getValueAsDouble();
        if (std::abs(angle - M_PI) <= ATOL) {
            is_adj = false;
        } else if (std::abs(angle - (-M_PI)) <= ATOL) {
            is_adj = true;
        } else {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();
        mlir::Value x = gate.getControls()[0];
        mlir::Value y = gate.getControls()[1];
        mlir::Value z = gate.getQubit();

        if (is_adj) {
            std::tie(x, y, z) = ccxphase_adj(rewriter, loc, x, y, z);
        } else {
            std::tie(x, y, z) = ccxphase(rewriter, loc, x, y, z);
        }

        rewriter.replaceOp(gate, {x, y, z});
        return mlir::success();
    }
};

struct DecomposeMultiControlPass : public qcirc::DecomposeMultiControlBase<DecomposeMultiControlPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceManyControlsPattern,
            ReplaceTweedledumCCXPhasePattern
        >(&getContext());

        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createDecomposeMultiControlPass() {
    return std::make_unique<DecomposeMultiControlPass>();
}
