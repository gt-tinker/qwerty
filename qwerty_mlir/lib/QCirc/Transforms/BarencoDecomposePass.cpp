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

// Generalized version of Figure 4.5 from Nielsen and Chuang
mlir::ValueRange globalPhase(mlir::Location loc,
                             mlir::OpBuilder &builder,
                             mlir::Value phase,
                             mlir::ValueRange controls) {
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

// Corollary 5.3 of Barenco et al. (1995)
void barenco(mlir::OpBuilder &builder,
             mlir::Location loc,
             EulerAngles &angles,
             mlir::ValueRange controls,
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

class ReplaceGate1QOpNoControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (!gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate1Q::X:
        case qcirc::Gate1Q::Y:
        case qcirc::Gate1Q::Z:
        case qcirc::Gate1Q::H:
        case qcirc::Gate1Q::S:
        case qcirc::Gate1Q::T:
            // These guys are fine
            return mlir::failure();

        // Rotate by -π around the Z axis and then rotate by π/2. This achieves
        // a rotation by -π/2. That is, SZ = Sdg.
        case qcirc::Gate1Q::Sdg: {
            mlir::Value z = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Z, mlir::ValueRange(), gate.getQubit()
                ).getResult();
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::S, mlir::ValueRange(), z);
            return mlir::success();
        }

        // Rotate by -π/2 around the Z axis and then rotate by π/4. This achieves
        // a rotation by -π/4. That is, TSdg = Tdg.
        case qcirc::Gate1Q::Tdg: {
            mlir::Value sdg = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(), gate.getQubit()
                ).getResult();
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::T, mlir::ValueRange(), sdg);
            return mlir::success();
        }

        case qcirc::Gate1Q::Sx: {
            // HSH = Sx
            mlir::Value h = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            mlir::Value s = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::S, mlir::ValueRange(), h
                ).getResult();
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::H, mlir::ValueRange(), s);
            return mlir::success();
        }
        case qcirc::Gate1Q::Sxdg: {
            // HSdgH = Sxdg
            mlir::Value h = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            mlir::Value sdg = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(), h
                ).getResult();
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::H, mlir::ValueRange(), sdg);
            return mlir::success();
        }
        default:
            assert(0 && "Missing gate in annoying gate replacement pass");
            return mlir::failure();
        }
    }
};

class ReplaceSxWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        // Only Sx is an issue here. Everyone else is fine.
        case qcirc::Gate1Q::X:
        case qcirc::Gate1Q::Y:
        case qcirc::Gate1Q::Z:
        case qcirc::Gate1Q::H:
        case qcirc::Gate1Q::S:
        case qcirc::Gate1Q::Sdg:
        case qcirc::Gate1Q::T:
        case qcirc::Gate1Q::Tdg:
            return mlir::failure();

        // Create a Hadmard-conjugated controlled-controlled-S and let
        // another iteration of this pattern take care of the CCS.
        case qcirc::Gate1Q::Sx: {
            mlir::Value x_to_z = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            qcirc::Gate1QOp ccs = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::S, gate.getControls(), x_to_z);
            mlir::Value z_to_x = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    ccs.getResult()
                ).getResult();
            llvm::SmallVector<mlir::Value> qubits(ccs.getControlResults());
            qubits.push_back(z_to_x);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        // Same as above for Sx, except the S is Sdg instead
        case qcirc::Gate1Q::Sxdg: {
            mlir::Value x_to_z = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            qcirc::Gate1QOp ccsdg = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg, gate.getControls(), x_to_z);
            mlir::Value z_to_x = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    ccsdg.getResult()
                ).getResult();
            llvm::SmallVector<mlir::Value> qubits(ccsdg.getControlResults());
            qubits.push_back(z_to_x);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        default:
            assert(0 && "Missing gate in annoying gate replacement pass");
            return mlir::failure();
        }
    }
};

class ReplaceGate1QOpWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate1Q::X:
            // -decompose-multi-control already dealt with any gates that
            // have more than 2 controls, so this must be a CX or CCX.
            if (gate.getControls().size() == 2) {
                // Equation (3) of Selinger (2013): https://doi.org/10.1103/PhysRevA.87.042302

                // First layer: I ⊗ I ⊗ H
                mlir::Value h1 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                        gate.getQubit()
                    ).getResult();

                // Second layer: Tdg ⊗ T ⊗ T
                mlir::Value tdg1 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(),
                        gate.getControls()[0]
                    ).getResult();
                mlir::Value t1 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::T, mlir::ValueRange(),
                        gate.getControls()[1]
                    ).getResult();
                mlir::Value t2 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::T, mlir::ValueRange(), h1
                    ).getResult();

                // Second layer: (|1⟩⟨1| ⊗ X + |0⟩⟨0| ⊗ I) ⊗ I
                qcirc::Gate1QOp cx1 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, tdg1, t1);
                assert(cx1.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx1_ctrl = cx1.getControlResults()[0];
                mlir::Value cx1_tgt = cx1.getResult();

                // Second layer: ((X ⊗ I) ⊗ |1⟩⟨1| + (I ⊗ I) ⊗ |0⟩⟨0|)
                qcirc::Gate1QOp cx2 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, t2, cx1_ctrl);
                assert(cx2.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx2_ctrl = cx2.getControlResults()[0];
                mlir::Value cx2_tgt = cx2.getResult();

                // Third layer: Tdg ⊗ (|1⟩⟨1| ⊗ X + |0⟩⟨0| ⊗ I)
                mlir::Value tdg2 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), cx2_tgt
                    ).getResult();

                qcirc::Gate1QOp cx3 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, cx1_tgt, cx2_ctrl);
                assert(cx3.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx3_ctrl = cx3.getControlResults()[0];
                mlir::Value cx3_tgt = cx3.getResult();

                // Fourth layer: (X ⊗ |1⟩⟨1| + I ⊗ |0⟩⟨0|) ⊗ I
                qcirc::Gate1QOp cx4 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, cx3_ctrl, tdg2);
                assert(cx4.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx4_ctrl = cx4.getControlResults()[0];
                mlir::Value cx4_tgt = cx4.getResult();

                // Fifth layer: Tdg ⊗ Tdg ⊗ T
                mlir::Value tdg3 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), cx4_tgt
                    ).getResult();
                mlir::Value tdg4 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), cx4_ctrl
                    ).getResult();
                mlir::Value t3 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::T, mlir::ValueRange(), cx3_tgt
                    ).getResult();

                // Sixth layer: ((X ⊗ I) ⊗ |1⟩⟨1| + (I ⊗ I) ⊗ |0⟩⟨0|)
                qcirc::Gate1QOp cx5 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, t3, tdg3);
                assert(cx5.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx5_ctrl = cx5.getControlResults()[0];
                mlir::Value cx5_tgt = cx5.getResult();

                // Seventh layer: S ⊗ (|1⟩⟨1| ⊗ X + |0⟩⟨0| ⊗ I)
                mlir::Value s = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::S, mlir::ValueRange(), cx5_tgt
                    ).getResult();

                qcirc::Gate1QOp cx6 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, tdg4, cx5_ctrl);
                assert(cx6.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx6_ctrl = cx6.getControlResults()[0];
                mlir::Value cx6_tgt = cx6.getResult();

                // Eighth layer: (|1⟩⟨1| ⊗ X + |0⟩⟨0| ⊗ I) ⊗ H
                qcirc::Gate1QOp cx7 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, s, cx6_ctrl);
                assert(cx7.getControlResults().size() == 1
                       && "Wrong number of controls");
                mlir::Value cx7_ctrl = cx7.getControlResults()[0];
                mlir::Value cx7_tgt = cx7.getResult();

                mlir::Value h2 = rewriter.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::H, mlir::ValueRange(), cx6_tgt
                    ).getResult();

                rewriter.replaceOp(gate, std::initializer_list<mlir::Value>{
                    cx7_ctrl, cx7_tgt, h2});
                return mlir::success();
            } else {
                // Preserve CXs and (if they somehow slipped through) CCCCCXs
                return mlir::failure();
            }

        case qcirc::Gate1Q::Y: {
            mlir::Value y_to_x = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            qcirc::Gate1QOp ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, gate.getControls(), y_to_x);
            mlir::Value x_to_y = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::S, mlir::ValueRange(),
                    ccx.getResult()
                ).getResult();
            llvm::SmallVector<mlir::Value> qubits(ccx.getControlResults());
            qubits.push_back(x_to_y);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        case qcirc::Gate1Q::Z: {
            mlir::Value z_to_x = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            qcirc::Gate1QOp ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, gate.getControls(), z_to_x);
            mlir::Value x_to_z = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(),
                    ccx.getResult()
                ).getResult();
            llvm::SmallVector<mlir::Value> qubits(ccx.getControlResults());
            qubits.push_back(x_to_z);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        // Suppose we want to accomplish a CCH. The overall idea is that on
        // the Bloch sphere, H is a rotation by π just as X is. So we
        // just need to translate the target qubit from the eigenbasis of H
        // to the eigenbasis of X, run the CCX, and then translate the
        // target back from the eigenbasis of X to the eigenbasis of H.
        // But what is the eigenbasis of H?
        // Let |H+⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩
        // and |H-⟩ = -sin(π/8)|0⟩ + cos(π/8)|1⟩.
        // It is easy to see that H|H+⟩ = |H+⟩ and H|H-⟩ = -|H-⟩ [1].
        // So we want an operator U = |+⟩⟨H+| +  |-⟩⟨H-|. In matrix form [1],
        // U = [ cos(π/8)  -sin(π/8) ]
        //     [ sin(π/8)   cos(π/8) ].
        // Obviously, U† X U = H. Now it is necessary to write U as
        // Clifford+T as follows:
        // U = X Ry(π/4)
        //   = X (e^(-iπ/8) S H T H S†).
        // Then we can substitute in U as follows:
        // H = U† X U
        //   = (e^(iπ/8) S H T† H S†) X X X (e^(-iπ/8) S H T H S†)
        //   = (S H T† H S†) X X X (S H T H S†)
        //   = (S H T† H S†) X (S H T H S†).
        // The last step still holds when the middle X is controlled.
        // [1]: https://meirizarrygelpi.github.io/posts/physics/hadamard-eigen-basis/index.html
        case qcirc::Gate1Q::H: {
            mlir::Value h_to_x1 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(),
                    gate.getQubit()
                ).getResult();
            mlir::Value h_to_x2 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(), h_to_x1
                ).getResult();
            mlir::Value h_to_x3 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::T, mlir::ValueRange(), h_to_x2
                ).getResult();
            mlir::Value h_to_x4 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(), h_to_x3
                ).getResult();
            mlir::Value h_to_x5 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::S, mlir::ValueRange(), h_to_x4
                ).getResult();

            qcirc::Gate1QOp ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, gate.getControls(), h_to_x5);

            mlir::Value x_to_h1 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(),
                    ccx.getResult()
                ).getResult();
            mlir::Value x_to_h2 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(), x_to_h1
                ).getResult();
            mlir::Value x_to_h3 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(), x_to_h2
                ).getResult();
            mlir::Value x_to_h4 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::H, mlir::ValueRange(), x_to_h3
                ).getResult();
            mlir::Value x_to_h5 = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::S, mlir::ValueRange(), x_to_h4
                ).getResult();

            llvm::SmallVector<mlir::Value> qubits(ccx.getControlResults());
            qubits.push_back(x_to_h5);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        // Let `ReplaceSxWithControlsPattern` above take care of this
        case qcirc::Gate1Q::Sx:
        case qcirc::Gate1Q::Sxdg:
            return mlir::failure();

        // Hand-tuned Barenco. The insight here is that because
        // XRz(θ)X = Rz(-θ), we have S = e^(iπ/4)XRz(-π/4)XRz(π/4).
        case qcirc::Gate1Q::S: {
            qcirc::Gate1QOp ccx1 = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, gate.getControls(), gate.getQubit());

            mlir::Value tdg = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(),
                    ccx1.getResult()
                ).getResult();

            qcirc::Gate1QOp ccx2 = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, ccx1.getControlResults(), tdg);

            mlir::Value t = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::T, mlir::ValueRange(),
                    ccx2.getResult()
                ).getResult();

            auto last_ctrl = ccx2.getControlResults().end();
            last_ctrl--;
            qcirc::Gate1QOp global_phase = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::T,
                llvm::iterator_range(ccx2.getControlResults().begin(),
                                     last_ctrl),
                *last_ctrl);

            llvm::SmallVector<mlir::Value> qubits(global_phase->getResults());
            qubits.push_back(t);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        // Very similar to S above
        case qcirc::Gate1Q::Sdg: {
            qcirc::Gate1QOp ccx1 = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, gate.getControls(), gate.getQubit());

            mlir::Value t = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::T, mlir::ValueRange(),
                    ccx1.getResult()
                ).getResult();

            qcirc::Gate1QOp ccx2 = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, ccx1.getControlResults(), t);

            mlir::Value tdg = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Tdg, mlir::ValueRange(),
                    ccx2.getResult()
                ).getResult();

            auto last_ctrl = ccx2.getControlResults().end();
            last_ctrl--;
            qcirc::Gate1QOp global_phase = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::Tdg,
                llvm::iterator_range(ccx2.getControlResults().begin(),
                                     last_ctrl),
                *last_ctrl);

            llvm::SmallVector<mlir::Value> qubits(global_phase->getResults());
            qubits.push_back(tdg);
            rewriter.replaceOp(gate, qubits);
            return mlir::success();
        }

        // Let's not bother with a hand-tuned Barenco decomposition because
        // there is not an easy way to do Rz(π/8) with Cliffords. Instead,
        // let's emit Rzs and let a downstream Solovay–Kitaev nerd sort it
        // out after getting shoved in their locker.
        case qcirc::Gate1Q::T:
        case qcirc::Gate1Q::Tdg: {
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
};

class ReplaceGate1Q1POpNoControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate, mlir::PatternRewriter &rewriter) const override {
        if (!gate.getControls().empty()) {
            return mlir::failure();
        }

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
    }
};

class ReplaceGate1Q1POpWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate1Q1P::Rx:
        case qcirc::Gate1Q1P::Ry:
        case qcirc::Gate1Q1P::Rz:
        case qcirc::Gate1Q1P::P: {
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
};

class ReplacePWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        // These are fine
        case qcirc::Gate1Q1P::Rx:
        case qcirc::Gate1Q1P::Ry:
        case qcirc::Gate1Q1P::Rz:
            return mlir::failure();

        case qcirc::Gate1Q1P::P: {
            mlir::Value const_2 = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getF64FloatAttr(2.0)).getResult();
            mlir::Value theta_div_2 = rewriter.create<mlir::arith::DivFOp>(
                loc, gate.getParam(), const_2).getResult();
            mlir::ValueRange next_controls = globalPhase(
                loc, rewriter, theta_div_2, gate.getControls());

            rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(
                gate, qcirc::Gate1Q1P::Rz, gate.getParam(), next_controls,
                gate.getQubit());
            return mlir::success();
       }

        default:
            assert(0 && "Missing gate in annoying gate replacement pass");
            return mlir::failure();
        }
    }
};

class ReplaceGate1Q3POpNoControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1Q3POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q3POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q3POp gate, mlir::PatternRewriter &rewriter) const override {
        if (!gate.getControls().empty()) {
            return mlir::failure();
        }

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
    }
};

class ReplaceGate1Q3POpWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1Q3POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q3POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q3POp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate1Q3P::U: {
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
};

class ReplaceUWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate1Q3POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q3POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q3POp gate, mlir::PatternRewriter &rewriter) const override {
        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate1Q3P::U: {
            mlir::Value theta = gate.getFirstParam();
            mlir::Value phi = gate.getSecondParam();
            mlir::Value lambda = gate.getThirdParam();

            mlir::Value theta_plus_lambda =
                rewriter.create<mlir::arith::AddFOp>(
                    loc, theta, lambda).getResult();
            mlir::Value two = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getF64FloatAttr(2)).getResult();
            mlir::Value theta_plus_lambda_div_2 =
                rewriter.create<mlir::arith::DivFOp>(
                    loc, theta_plus_lambda, two).getResult();
            mlir::ValueRange next_controls = globalPhase(
                loc, rewriter, theta_plus_lambda_div_2, gate.getControls());

            qcirc::Gate1Q1POp rz_lambda = rewriter.create<qcirc::Gate1Q1POp>(
                loc, qcirc::Gate1Q1P::Rz, lambda, next_controls,
                gate.getQubit());
            qcirc::Gate1Q1POp ry_phi = rewriter.create<qcirc::Gate1Q1POp>(
                loc, qcirc::Gate1Q1P::Ry, phi, rz_lambda.getControlResults(),
                rz_lambda.getResult());
            // rz_theta
            rewriter.replaceOpWithNewOp<qcirc::Gate1Q1POp>(
                gate, qcirc::Gate1Q1P::Rz, theta, ry_phi.getControlResults(),
                ry_phi.getResult());
            return mlir::success();
        }

        default:
            assert(0 && "Missing gate in annoying gate replacement pass");
            return mlir::failure();
        }
    }
};

class ReplaceGate2QOpNoControlsPattern : public mlir::OpRewritePattern<qcirc::Gate2QOp> {
    using mlir::OpRewritePattern<qcirc::Gate2QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate2QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (!gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate2Q::Swap: {
            qcirc::Gate1QOp cnot1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, gate.getLeftQubit(), gate.getRightQubit());
            qcirc::Gate1QOp cnot2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, cnot1.getResult(), cnot1.getControlResults()[0]);
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::X, cnot2.getResult(), cnot2.getControlResults()[0]);
            return mlir::success();
        }

        default:
            assert(0 && "Missing gate in annoying gate replacement pass");
            return mlir::failure();
        }
    }
};

class ReplaceGate2QOpWithControlsPattern : public mlir::OpRewritePattern<qcirc::Gate2QOp> {
    using mlir::OpRewritePattern<qcirc::Gate2QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate2QOp gate, mlir::PatternRewriter &rewriter) const override {
        if (gate.getControls().empty()) {
            return mlir::failure();
        }

        mlir::Location loc = gate.getLoc();

        switch (gate.getGate()) {
        case qcirc::Gate2Q::Swap: {
            llvm::SmallVector<mlir::Value> cnot1_controls(gate.getControls().begin(), gate.getControls().end());
            cnot1_controls.push_back(gate.getLeftQubit());
            qcirc::Gate1QOp cnot1 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, cnot1_controls, gate.getRightQubit());

            auto second_to_last = cnot1.getControlResults().end();
            second_to_last -= 1;
            llvm::SmallVector<mlir::Value> cnot2_controls(cnot1.getControlResults().begin(), second_to_last);
            cnot2_controls.push_back(cnot1.getResult());
            mlir::Value cnot2_target = cnot1.getControlResults()[cnot1.getControlResults().size()-1];
            qcirc::Gate1QOp cnot2 = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, cnot2_controls, cnot2_target);

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
};

// This preserves controls. These patterns must not do any Barenco
// decompositions.
struct ReplaceUnusualGatesPass : public qcirc::ReplaceUnusualGatesBase<ReplaceUnusualGatesPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceGate1QOpNoControlsPattern,
            ReplaceSxWithControlsPattern,
            ReplaceGate1Q1POpNoControlsPattern,
            ReplacePWithControlsPattern,
            ReplaceGate1Q3POpNoControlsPattern,
            ReplaceUWithControlsPattern,
            ReplaceGate2QOpNoControlsPattern,
            ReplaceGate2QOpWithControlsPattern
        >(&getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

// This preserves controls only on X gates. It performs Barenco decompositions
// to accomplish that.
struct BarencoDecomposePass : public qcirc::BarencoDecomposeBase<BarencoDecomposePass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceGate1QOpNoControlsPattern,
            ReplaceSxWithControlsPattern,
            ReplaceGate1QOpWithControlsPattern,
            ReplaceGate1Q1POpNoControlsPattern,
            ReplaceGate1Q1POpWithControlsPattern,
            ReplaceGate1Q3POpNoControlsPattern,
            ReplaceGate1Q3POpWithControlsPattern,
            ReplaceGate2QOpNoControlsPattern,
            ReplaceGate2QOpWithControlsPattern
        >(&getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createReplaceUnusualGatesPass() {
    return std::make_unique<ReplaceUnusualGatesPass>();
}

std::unique_ptr<mlir::Pass> qcirc::createBarencoDecomposePass() {
    return std::make_unique<BarencoDecomposePass>();
}
