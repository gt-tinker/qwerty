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

// Mission: get rid of any gates that will probably cause a headache when we
// lower to QIR. We assume _ctl exists (otherwise: here's a nickel son, buy
// yourself a better computer), so the mission is to convert gates _not_ in this
// (apparently highly tentative) list into gates that _are_ on this list:
// https://github.com/qir-alliance/qir-spec/blob/365efe3d19414812a7e5e2032aceaca8d0742347/specification/under_development/Instruction_Set.md

namespace {

// Generalized version of Figure 4.5 from Nielsen and Chuang
mlir::ValueRange globalPhase(mlir::Location loc, mlir::OpBuilder &builder, mlir::Value phase, mlir::ValueRange controls) {
    assert(!controls.empty() && "Cannot apply a global phase to nothing, brother!");
    auto second_to_last = controls.end();
    second_to_last -= 1;
    llvm::SmallVector<mlir::Value> phase_controls(controls.begin(), second_to_last);
    // New control inputs for the adjusted gate
    return builder.create<qcirc::Gate1Q1POp>(
            loc,
            qcirc::Gate1Q1P::P,
            phase,
            phase_controls,
            controls[controls.size()-1])->getResults();
}

class ReplaceGate1QOpPattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            switch (gate.getGate()) {
            case qcirc::Gate1Q::X:
            case qcirc::Gate1Q::Y:
            case qcirc::Gate1Q::Z:
            case qcirc::Gate1Q::H:
            case qcirc::Gate1Q::S:
            case qcirc::Gate1Q::Sdg:
            case qcirc::Gate1Q::T:
            case qcirc::Gate1Q::Tdg:
                // These guys are fine
                return mlir::failure();

            case qcirc::Gate1Q::Sx: {
                // Global phase is ok because this is not controlled
                mlir::Value pi_div_2 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(M_PI_2)).getResult();
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rx, pi_div_2, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            }
            case qcirc::Gate1Q::Sxdg: {
                // Global phase is ok because this is not controlled
                mlir::Value neg_pi_div_2 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(-M_PI_2)).getResult();
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rx, neg_pi_div_2, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        } else { // Has controls
            switch (gate.getGate()) {
            case qcirc::Gate1Q::X:
            case qcirc::Gate1Q::Y:
            case qcirc::Gate1Q::Z:
            case qcirc::Gate1Q::H:
            case qcirc::Gate1Q::S:
            case qcirc::Gate1Q::Sdg:
            case qcirc::Gate1Q::T:
            case qcirc::Gate1Q::Tdg:
                // These guys are still fine
                return mlir::failure();

            case qcirc::Gate1Q::Sx: {
                mlir::Value pi_div_4 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(M_PI_4)).getResult();
                mlir::ValueRange next_controls = globalPhase(gate.getLoc(), rewriter, pi_div_4, gate.getControls());

                mlir::Value pi_div_2 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(M_PI_2)).getResult();
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rx, pi_div_2, next_controls, gate.getQubit());
                return mlir::success();
            }
            case qcirc::Gate1Q::Sxdg: {
                mlir::Value neg_pi_div_4 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(-M_PI_4)).getResult();
                mlir::ValueRange next_controls = globalPhase(gate.getLoc(), rewriter, neg_pi_div_4, gate.getControls());

                mlir::Value neg_pi_div_2 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(-M_PI_2)).getResult();
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rx, neg_pi_div_2, next_controls, gate.getQubit());
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        }
    }
};

class ReplaceGate1Q1POpPattern : public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            switch (gate.getGate()) {
            case qcirc::Gate1Q1P::Rx:
            case qcirc::Gate1Q1P::Ry:
            case qcirc::Gate1Q1P::Rz:
                // These guys are fine
                return mlir::failure();

            case qcirc::Gate1Q1P::P: {
                // Global phase is ok because this is not controlled
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rz, gate.getParam(), mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        } else { // Has controls
            switch (gate.getGate()) {
            case qcirc::Gate1Q1P::Rx:
            case qcirc::Gate1Q1P::Ry:
            case qcirc::Gate1Q1P::Rz:
                // These guys are still fine
                return mlir::failure();

            case qcirc::Gate1Q1P::P: {
                mlir::Value const_2 = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(2.0)).getResult();
                mlir::Value theta_div_2 = rewriter.create<mlir::arith::DivFOp>(gate.getLoc(), gate.getParam(), const_2).getResult();
                mlir::ValueRange next_controls = globalPhase(gate.getLoc(), rewriter, theta_div_2, gate.getControls());

                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rz, gate.getParam(), next_controls, gate.getQubit());
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        }
    }
};


class ReplaceGate1Q3POpPattern : public mlir::OpRewritePattern<qcirc::Gate1Q3POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q3POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q3POp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            switch (gate.getGate()) {
            case qcirc::Gate1Q3P::U: {
                // Global phase is ok because this is not controlled
                mlir::Value rz_lambda = rewriter.create<qcirc::Gate1Q1POp>(gate.getLoc(), qcirc::Gate1Q1P::Rz, gate.getThirdParam(), mlir::ValueRange(), gate.getQubit()).getResult();
                mlir::Value ry_phi = rewriter.create<qcirc::Gate1Q1POp>(gate.getLoc(), qcirc::Gate1Q1P::Ry, gate.getSecondParam(), mlir::ValueRange(), rz_lambda).getResult();
                // rz_theta
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rz, gate.getFirstParam(), mlir::ValueRange(), ry_phi);
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        } else { // Has controls
            switch (gate.getGate()) {
            case qcirc::Gate1Q3P::U: {
                mlir::Value theta_plus_lambda = rewriter.create<mlir::arith::AddFOp>(gate.getLoc(), gate.getFirstParam(), gate.getThirdParam()).getResult();
                mlir::Value two = rewriter.create<mlir::arith::ConstantOp>(gate.getLoc(), rewriter.getF64FloatAttr(2)).getResult();
                mlir::Value theta_plus_lambda_div_2 = rewriter.create<mlir::arith::DivFOp>(gate.getLoc(), theta_plus_lambda, two).getResult();
                mlir::ValueRange next_controls = globalPhase(gate.getLoc(), rewriter, theta_plus_lambda_div_2, gate.getControls());

                qcirc::Gate1Q1POp rz_lambda = rewriter.create<qcirc::Gate1Q1POp>(gate.getLoc(), qcirc::Gate1Q1P::Rz, gate.getThirdParam(), next_controls, gate.getQubit());
                qcirc::Gate1Q1POp ry_phi = rewriter.create<qcirc::Gate1Q1POp>(gate.getLoc(), qcirc::Gate1Q1P::Ry, gate.getSecondParam(), rz_lambda.getControlResults(), rz_lambda.getResult());
                // rz_theta
                rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(gate, qcirc::Gate1Q1P::Rz, gate.getFirstParam(), ry_phi.getControlResults(), ry_phi.getResult());
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        }
    }
};

class ReplaceGate2QOpPattern : public mlir::OpRewritePattern<qcirc::Gate2QOp> {
    using mlir::OpRewritePattern<qcirc::Gate2QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate2QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            switch (gate.getGate()) {
            case qcirc::Gate2Q::Swap: {
                qcirc::Gate1QOp cnot1 = rewriter.create<qcirc::Gate1QOp>(gate.getLoc(), qcirc::Gate1Q::X, gate.getLeftQubit(), gate.getRightQubit());
                qcirc::Gate1QOp cnot2 = rewriter.create<qcirc::Gate1QOp>(gate.getLoc(), qcirc::Gate1Q::X, cnot1.getResult(), cnot1.getControlResults()[0]);
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::X, cnot2.getResult(), cnot2.getControlResults()[0]);
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        } else { // Has controls
            switch (gate.getGate()) {
            case qcirc::Gate2Q::Swap: {
                llvm::SmallVector<mlir::Value> cnot1_controls(gate.getControls().begin(), gate.getControls().end());
                cnot1_controls.push_back(gate.getLeftQubit());
                qcirc::Gate1QOp cnot1 = rewriter.create<qcirc::Gate1QOp>(gate.getLoc(), qcirc::Gate1Q::X, cnot1_controls, gate.getRightQubit());

                auto second_to_last = cnot1.getControlResults().end();
                second_to_last -= 1;
                llvm::SmallVector<mlir::Value> cnot2_controls(cnot1.getControlResults().begin(), second_to_last);
                cnot2_controls.push_back(cnot1.getResult());
                mlir::Value cnot2_target = cnot1.getControlResults()[cnot1.getControlResults().size()-1];
                qcirc::Gate1QOp cnot2 = rewriter.create<qcirc::Gate1QOp>(gate.getLoc(), qcirc::Gate1Q::X, cnot2_controls, cnot2_target);

                second_to_last = cnot2.getControlResults().end();
                second_to_last -= 1;
                llvm::SmallVector<mlir::Value> cnot3_controls(cnot2.getControlResults().begin(), second_to_last);
                cnot3_controls.push_back(cnot2.getResult());
                mlir::Value cnot3_target = cnot2.getControlResults()[cnot2.getControlResults().size()-1];
                // cnot3
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::X, cnot3_controls, cnot3_target);
                return mlir::success();
            }

            default:
                assert(0 && "Missing gate in annoying gate replacement pass");
                return mlir::failure();
            }
        }
    }
};

struct ReplaceAnnoyingGatesPass : public qcirc::ReplaceAnnoyingGatesBase<ReplaceAnnoyingGatesPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceGate1QOpPattern,
            ReplaceGate1Q1POpPattern,
            ReplaceGate1Q3POpPattern,
            ReplaceGate2QOpPattern
        >(&getContext());

        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createReplaceAnnoyingGatesPass() {
    return std::make_unique<ReplaceAnnoyingGatesPass>();
}
