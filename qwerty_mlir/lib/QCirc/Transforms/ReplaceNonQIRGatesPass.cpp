// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include <variant>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "PassDetail.h"

// Mission: get rid of any gates that will probably cause a headache when we
// lower to QIR. The mission is to convert gates _not_ in this (apparently
// highly tentative) list into gates that _are_ on this list:
// https://github.com/qir-alliance/qir-spec/blob/365efe3d19414812a7e5e2032aceaca8d0742347/specification/under_development/Instruction_Set.md
// We do not assume that __ctl exists, since it often does not for the base
// profile.

namespace {

// "Son, we have Rust enums at home..."
struct Angle {
    using Storage = std::variant<double, mlir::Value>;

    Storage storage;

    Angle(const Angle &other) = default;
    Angle(Angle &&other) = default;
    Angle(double theta) : storage(theta) {}
    Angle(mlir::Value value) : storage(value) {}

    // Try to get the most explicit form of this angle as we can.
    // You can pass either an mlir::Value or double to this.
    static Angle get(Storage theta) {
        if (std::holds_alternative<double>(theta)) {
            return std::get<double>(theta);
        } else {
            assert(std::holds_alternative<mlir::Value>(theta)
                   && "angle must be either double or mlir::Value");
            mlir::Value theta_val = std::get<mlir::Value>(theta);

            mlir::FloatAttr theta_attr;
            if (mlir::matchPattern(theta_val, qcirc::m_CalcConstant(&theta_attr))) {
                return theta_attr.getValueAsDouble();
            } else {
                return std::get<mlir::Value>(theta);
            }
        }
    }

    bool isZero() {
        return std::holds_alternative<double>(storage)
               && std::abs(std::get<double>(storage)) < ATOL;
    }

    mlir::Value asValue(mlir::OpBuilder &builder, mlir::Location loc) {
        if (std::holds_alternative<mlir::Value>(storage)) {
            return std::get<mlir::Value>(storage);
        } else {
            return qcirc::stationaryF64Const(builder, loc,
                                             std::get<double>(storage));
        }
    }

    Angle unaryOp(mlir::OpBuilder &builder, mlir::Location loc,
                  std::function<double(double)> const_cb,
                  std::function<mlir::Value(mlir::Value)> val_cb) {
        if (std::holds_alternative<double>(storage)) {
            return const_cb(std::get<double>(storage));
        } else {
            assert(std::holds_alternative<mlir::Value>(storage)
                   && "angle must be either double or mlir::Value");
            return val_cb(std::get<mlir::Value>(storage));
        }
    }

    static Angle binaryOp(
            mlir::OpBuilder &builder, mlir::Location loc,
            Angle &left, Angle &right,
            std::function<double(double, double)> const_cb,
            std::function<mlir::Value(mlir::Value, mlir::Value)> val_cb) {
        if (std::holds_alternative<double>(left.storage)
                && std::holds_alternative<double>(right.storage)) {
            return const_cb(std::get<double>(left.storage),
                            std::get<double>(right.storage));
        } else {
            mlir::Value left_val = left.asValue(builder, loc);
            mlir::Value right_val = right.asValue(builder, loc);

            return qcirc::wrapStationaryF64Ops(builder, loc,
                std::initializer_list<mlir::Value>{left_val, right_val},
                [&](mlir::ValueRange args) {
                    assert(args.size() == 2);
                    mlir::Value left_arg = args[0];
                    mlir::Value right_arg = args[1];
                    return val_cb(left_arg, right_arg);
                });
        }
    }
};

// e^{iϕ}R_z(α)R_y(θ)R_z(β)
struct EulerAngles {
    Angle phi;
    Angle alpha;
    Angle theta;
    Angle beta;

    EulerAngles(Angle phi, Angle alpha, Angle theta, Angle beta)
               : phi(phi), alpha(alpha), theta(theta), beta(beta) {}

    static EulerAngles forGate(qcirc::Gate1QOp gate) {
        switch (gate.getGate()) {
        //                                               ϕ        α         θ       β
        case qcirc::Gate1Q::X:    return EulerAngles(   M_PI_2, -M_PI_2,    M_PI,  M_PI_2);
        case qcirc::Gate1Q::Y:    return EulerAngles(   M_PI_2,     0.0,    M_PI,     0.0);
        case qcirc::Gate1Q::Z:    return EulerAngles(   M_PI_2,     0.0,     0.0,    M_PI);
        case qcirc::Gate1Q::H:    return EulerAngles(   M_PI_2,     0.0,  M_PI_2,    M_PI);
        case qcirc::Gate1Q::S:    return EulerAngles(   M_PI_4,     0.0,     0.0,  M_PI_2);
        case qcirc::Gate1Q::Sdg:  return EulerAngles(   M_PI_4,     0.0,     0.0, -M_PI_2);
        case qcirc::Gate1Q::T:    return EulerAngles( M_PI/8.0,     0.0,     0.0,  M_PI_4);
        case qcirc::Gate1Q::Tdg:  return EulerAngles(-M_PI/8.0,     0.0,     0.0, -M_PI_4);
        case qcirc::Gate1Q::Sx:   return EulerAngles(   M_PI_4, -M_PI_2,  M_PI_2,  M_PI_2);
        case qcirc::Gate1Q::Sxdg: return EulerAngles(  -M_PI_4, -M_PI_2, -M_PI_2,  M_PI_2);
        default:
            assert(0 && "Missing 1Q gate Euler angles");
            return EulerAngles(0.0, 0.0, 0.0, 0.0);
        }
    }

    static EulerAngles forGate(mlir::OpBuilder &builder, qcirc::Gate1Q1POp gate) {
        Angle param = Angle::get(gate.getParam());

        switch (gate.getGate()) {
        //                                                 ϕ              α         θ       β
        case qcirc::Gate1Q1P::Rx: return EulerAngles(            0.0, -M_PI_2,    param,  M_PI_2);
        case qcirc::Gate1Q1P::Ry: return EulerAngles(            0.0,     0.0,    param,     0.0);
        case qcirc::Gate1Q1P::Rz: return EulerAngles(            0.0,     0.0,      0.0,   param);
        case qcirc::Gate1Q1P::P: {
            mlir::Location loc = gate.getLoc();
            Angle theta_over_2 = param.unaryOp(
                builder, loc,
                [](double theta) { return theta/2.0; },
                [&](mlir::Value theta) {
                    mlir::Value const_2 = builder.create<mlir::arith::ConstantOp>(
                            loc, builder.getF64FloatAttr(2.0)).getResult();
                    return builder.create<mlir::arith::DivFOp>(
                        loc, theta, const_2).getResult();
                });
            //                      ϕ        α  θ    β
            return EulerAngles(theta_over_2, 0, 0, param);
        }
        default:
            assert(0 && "Missing 1Q1P gate Euler angles");
            return EulerAngles(0.0, 0.0, 0.0, 0.0);
        }
    }

    static EulerAngles forGate(mlir::OpBuilder &builder, qcirc::Gate1Q3POp gate) {
        mlir::Location loc = gate.getLoc();
        Angle u_theta = Angle::get(gate.getFirstParam());
        Angle u_phi = Angle::get(gate.getSecondParam());
        Angle u_lambda = Angle::get(gate.getThirdParam());

        switch (gate.getGate()) {
        // Equation (2) of Cross et al. 2022
        case qcirc::Gate1Q3P::U: {
            Angle phi_plus_lambda_over_2 = Angle::binaryOp(
                builder, loc, u_phi, u_lambda,
                [](double phi, double lambda) { return (phi + lambda)/2.0; },
                [&](mlir::Value phi, mlir::Value lambda) {
                    mlir::Value sum = builder.create<mlir::arith::AddFOp>(
                        loc, phi, lambda).getResult();
                    mlir::Value const_2 = builder.create<mlir::arith::ConstantOp>(
                            loc, builder.getF64FloatAttr(2.0)).getResult();
                    return builder.create<mlir::arith::DivFOp>(
                        loc, sum, const_2).getResult();
                });
            return EulerAngles(phi_plus_lambda_over_2, u_phi, u_theta, u_lambda);
        }
        default:
            assert(0 && "Missing 1Q3P gate Euler angles");
            return EulerAngles(0.0, 0.0, 0.0, 0.0);
        }
    }
};

// Corollary 5.3 of Barenco et al. (1995)
void barenco(mlir::OpBuilder &builder, mlir::Location loc,
             EulerAngles &angles, mlir::ValueRange controls,
             mlir::Value target,
             llvm::SmallVectorImpl<mlir::Value> &qubits_out) {
    llvm::SmallVector<mlir::Value> control_qubits(controls);

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    //                                           ^^^^^^^^^^^^^

    Angle beta_minus_alpha_over_2 = Angle::binaryOp(
        builder, loc, angles.beta, angles.alpha,
        [](double beta, double alpha) {
            return (beta - alpha) / 2.0;
        },
        [&](mlir::Value beta, mlir::Value alpha) {
            mlir::Value beta_minus_alpha = builder.create<mlir::arith::SubFOp>(
                loc, beta, alpha).getResult();
            mlir::Value const_2 = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getF64FloatAttr(2.0)).getResult();
            return builder.create<mlir::arith::DivFOp>(
                loc, beta_minus_alpha, const_2).getResult();
        });

    if (!beta_minus_alpha_over_2.isZero()) {
        mlir::Value beta_minus_alpha_over_2_val =
            beta_minus_alpha_over_2.asValue(builder, loc);
        qcirc::Gate1Q1POp rz1 = builder.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::Rz, beta_minus_alpha_over_2_val,
            mlir::ValueRange(), target);
        assert(rz1.getControlResults().empty());
        target = rz1.getResult();
    }

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    //                                         ^
    qcirc::Gate1QOp cnot1 = builder.create<qcirc::Gate1QOp>(
        loc, qcirc::Gate1Q::X, control_qubits, target);
    assert(cnot1.getControlResults().size() == control_qubits.size());
    control_qubits = cnot1.getControlResults();
    //control_qubits.clear();
    //control_qubits.append(cnot1.getControlResults().begin(),
    //                      cnot1.getControlResults().end());
    target = cnot1.getResult();

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    //                          ^^^^^^^^^^^^^^
    Angle neg_alpha_plus_beta_over_2 = Angle::binaryOp(
        builder, loc, angles.alpha, angles.beta,
        [](double alpha, double beta) {
            return -(alpha + beta) / 2.0;
        },
        [&](mlir::Value alpha, mlir::Value beta) {
            mlir::Value alpha_plus_beta = builder.create<mlir::arith::AddFOp>(
                loc, alpha, beta).getResult();
            mlir::Value const_2 = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getF64FloatAttr(2.0)).getResult();
            mlir::Value div_by_2 = builder.create<mlir::arith::DivFOp>(
                loc, alpha_plus_beta, const_2).getResult();
            return builder.create<mlir::arith::NegFOp>(
                loc, div_by_2).getResult();
        });

    if (!neg_alpha_plus_beta_over_2.isZero()) {
        mlir::Value neg_alpha_plus_beta_over_2_val =
            neg_alpha_plus_beta_over_2.asValue(builder, loc);
        qcirc::Gate1Q1POp rz2 = builder.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::Rz, neg_alpha_plus_beta_over_2_val,
            mlir::ValueRange(), target);
        assert(rz2.getControlResults().empty());
        target = rz2.getResult();
    }

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    //                 ^^^^^^^^
    Angle neg_theta_over_2 = angles.theta.unaryOp(
        builder, loc,
        [](double theta) {
            return -theta/2.0;
        },
        [&](mlir::Value theta) {
            mlir::Value const_2 = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getF64FloatAttr(2.0)).getResult();
            mlir::Value div_by_2 = builder.create<mlir::arith::DivFOp>(
                loc, theta, const_2).getResult();
            return builder.create<mlir::arith::NegFOp>(
                loc, div_by_2).getResult();
        });

    if (!neg_theta_over_2.isZero()) {
        mlir::Value neg_theta_over_2_val = neg_theta_over_2.asValue(
            builder, loc);
        qcirc::Gate1Q1POp ry1 = builder.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::Ry, neg_theta_over_2_val, mlir::ValueRange(),
            target);
        assert(ry1.getControlResults().empty());
        target = ry1.getResult();
    }

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    //               ^
    qcirc::Gate1QOp cnot2 = builder.create<qcirc::Gate1QOp>(
        loc, qcirc::Gate1Q::X, control_qubits, target);
    assert(cnot2.getControlResults().size() == control_qubits.size());
    control_qubits = cnot2.getControlResults();
    //control_qubits.clear();
    //control_qubits.append(cnot2.getControlResults().begin(),
    //                      cnot2.getControlResults().end());
    target = cnot2.getResult();

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    //       ^^^^^^^
    Angle theta_over_2 = angles.theta.unaryOp(
        builder, loc,
        [](double theta) {
            return theta/2.0;
        },
        [&](mlir::Value theta) {
            mlir::Value const_2 = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getF64FloatAttr(2.0)).getResult();
            return builder.create<mlir::arith::DivFOp>(
                loc, theta, const_2).getResult();
        });

    if (!theta_over_2.isZero()) {
        mlir::Value theta_over_2_val = theta_over_2.asValue(builder, loc);
        qcirc::Gate1Q1POp ry2 = builder.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::Ry, theta_over_2_val, mlir::ValueRange(), target);
        assert(ry2.getControlResults().empty());
        target = ry2.getResult();
    }

    // Rz(α) Ry(θ/2) X Ry(-θ/2) Rz(-(α + β)/2) X Rz((β - α)/2)
    // ^^^^^
    if (!angles.alpha.isZero()) {
        mlir::Value alpha_val = angles.alpha.asValue(builder, loc);
        qcirc::Gate1Q1POp rz2 = builder.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::Rz, alpha_val, mlir::ValueRange(),
            target);
        assert(rz2.getControlResults().empty());
        target = rz2.getResult();
    }

    // Now need to deal with global phase
    if (!angles.phi.isZero()) {
        mlir::Value phi_val = angles.phi.asValue(builder, loc);
        qcirc::Gate1Q1POp cp = builder.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::P, phi_val,
            llvm::iterator_range(control_qubits.begin(), control_qubits.end()-1),
            *(control_qubits.end()-1));
        control_qubits = cp.getControlResults();
        control_qubits.push_back(cp.getResult());
    }

    qubits_out.clear();
    qubits_out.append(control_qubits);
    qubits_out.push_back(target);
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
                // -decompose-multi-control already dealt with any gates that
                // have more than 2 controls, so this must be a CX or CCX.
                return mlir::failure();

            case qcirc::Gate1Q::Y:
            case qcirc::Gate1Q::Z:
            case qcirc::Gate1Q::H:
            case qcirc::Gate1Q::S:
            case qcirc::Gate1Q::Sdg:
            case qcirc::Gate1Q::T:
            case qcirc::Gate1Q::Tdg:
            case qcirc::Gate1Q::Sx:
            case qcirc::Gate1Q::Sxdg: {
                mlir::Location loc = gate.getLoc();
                EulerAngles angles = EulerAngles::forGate(gate);
                llvm::SmallVector<mlir::Value> barenco_qubits;
                barenco(rewriter, loc, angles, gate.getControls(),
                        gate.getQubit(), barenco_qubits);
                rewriter.replaceOp(gate, barenco_qubits);
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
            case qcirc::Gate1Q1P::P: {
                mlir::Location loc = gate.getLoc();
                EulerAngles angles = EulerAngles::forGate(rewriter, gate);
                llvm::SmallVector<mlir::Value> barenco_qubits;
                barenco(rewriter, loc, angles, gate.getControls(),
                        gate.getQubit(), barenco_qubits);
                rewriter.replaceOp(gate, barenco_qubits);
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
                mlir::Location loc = gate.getLoc();
                EulerAngles angles = EulerAngles::forGate(rewriter, gate);
                llvm::SmallVector<mlir::Value> barenco_qubits;
                barenco(rewriter, loc, angles, gate.getControls(),
                        gate.getQubit(), barenco_qubits);
                rewriter.replaceOp(gate, barenco_qubits);
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

struct ReplaceNonQIRGatesPass : public qcirc::ReplaceNonQIRGatesBase<ReplaceNonQIRGatesPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceGate1QOpPattern,
            ReplaceGate1Q1POpPattern,
            ReplaceGate1Q3POpPattern,
            ReplaceGate2QOpPattern
        >(&getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createReplaceNonQIRGatesPass() {
    return std::make_unique<ReplaceNonQIRGatesPass>();
}
