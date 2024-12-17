// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp" // ATOL, M_PI
#include <unordered_set>
#include <vector>
#include <set>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tweedledum/Utils/Numbers.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "PassDetail.h"

// This pass does some gate-level peephole optimizations as described in
// Section 6.5 of the CGO paper. This pass is written assuming that it runs
// after all function specializations have already been generated — that is,
// it pays no attention to global phase. That is, if this pass runs on code
// that will later be predicated/controlled, the result may be incorrect.
// (The current compiler implementation does not do this, only running this
// pass right before lowering to QIR, after all function specializations are
// already generated.)

namespace {

// Replace controlled-X gates where the target is a |-⟩ with controlled-Z
// gates. This optimization is due to Liu, Bello, and Zhou (2021):
// https://doi.org/10.1109/CGO51591.2021.9370310. See Figure 10 of the CGO
// paper for a drawing. This optimization is useful because such code is
// commonly generated by writing f.phase.
class ReplaceZPattern : public mlir::OpRewritePattern<qcirc::QfreeOp> {
    using mlir::OpRewritePattern<qcirc::QfreeOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::QfreeOp gate, mlir::PatternRewriter &rewriter) const override {
        std::vector<mlir::Operation *> deleteGates;
        std::vector<qcirc::Gate1QOp> extraGates;
        std::vector<qcirc::Gate1QOp> xGates;
        deleteGates.push_back(gate);

        // Check if gates from QfreeZero are single-qubit gates
        mlir::Value qubit = gate.getQubit();
        qcirc::Gate1QOp gate1Q = qubit.getDefiningOp<qcirc::Gate1QOp>();
        if(!gate1Q) {
            return mlir::failure();
        }

        // Collect the single-qubit gates working on |->
        while(gate1Q.getControls().size() == 0) {
            extraGates.push_back(gate1Q);
            qubit = gate1Q.getQubit();

            gate1Q = qubit.getDefiningOp<qcirc::Gate1QOp>();
            if(!gate1Q) {
                return mlir::failure();
            }
        }

        // Check if after single-qubit gates are X gates with controls
        qcirc::Gate1QOp gateX = gate1Q;
        if (!gateX) {
            return mlir::failure();
        }

        // Check how many X gates before we see H(X(QAlloc))
        while (gateX.getGate() == qcirc::Gate1Q::X) {
            // X gates need controls
            if(gateX.getControls().size() == 0) {
                return mlir::failure();
            }
            xGates.push_back(gateX);
            qubit = gateX.getQubit();

            gateX = qubit.getDefiningOp<qcirc::Gate1QOp>();
            if(!gateX) {
                return mlir::failure();
            }
        }

        // Check for the next gates of H(X(QAlloc))
        qcirc::Gate1QOp gateH = gateX;
        deleteGates.push_back(gateH);
        if(!gateH || gateH.getGate() != qcirc::Gate1Q::H) {
            return mlir::failure();
        }

        mlir::Value qubitH = gateH.getQubit();
        qcirc::Gate1QOp gateHX = qubitH.getDefiningOp<qcirc::Gate1QOp>();
        deleteGates.push_back(gateHX);
        if(!gateHX || gateHX.getGate() != qcirc::Gate1Q::X) {
            return mlir::failure();
        }

        mlir::Value qubitHX = gateHX.getQubit();
        qcirc::QallocOp gateAlloc = qubitHX.getDefiningOp<qcirc::QallocOp>();
        deleteGates.push_back(gateAlloc);
        if(!gateAlloc) {
            return mlir::failure();
        }

        // Replace the X gates with Z gates
        for(qcirc::Gate1QOp gateX : xGates) {
            mlir::Value controlQubit = gateX.getControls()[gateX.getControls().size()-1];
            auto secondLast = gateX.getControls().end();
            secondLast -= 1;
            llvm::SmallVector<mlir::Value> tempVector(gateX.getControls().begin(), secondLast);
            // Insertion point needs to be moved before every replacement instruction
            rewriter.setInsertionPoint(gateX);
            qcirc::Gate1QOp gateZ = rewriter.create<qcirc::Gate1QOp>(gateX.getLoc(), qcirc::Gate1Q::Z, tempVector, controlQubit);

            llvm::SmallVector<mlir::Value> newValues(gateZ->result_begin(), gateZ->result_end());
            newValues.push_back(gateX.getQubit());
            rewriter.replaceOp(gateX, newValues);
        }

        // Push the IR inputs through all extra single-qubits to get rid of them
        for(qcirc::Gate1QOp gate1Q : extraGates) {
            rewriter.replaceOp(gate1Q, gate1Q.getOperands());
        }

        // Erase the qfree(H(X(qalloc))) subsequent gates
        for(auto deleteGate : deleteGates) {
            rewriter.eraseOp(deleteGate);
        }
        return mlir::success();
    }

};

// Same as above, except matching on QfreeZero ops instead of Qfree ops. (The
// identity holds in either case.)
class ReplaceZZeroPattern : public mlir::OpRewritePattern<qcirc::QfreeZeroOp> {
    using mlir::OpRewritePattern<qcirc::QfreeZeroOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::QfreeZeroOp gate, mlir::PatternRewriter &rewriter) const override {
        std::vector<mlir::Operation *> deleteGates;
        std::vector<qcirc::Gate1QOp> extraGates;
        std::vector<qcirc::Gate1QOp> xGates;
        deleteGates.push_back(gate);

        // Check if gates from QfreeZero are single-qubit gates
        mlir::Value qubit = gate.getQubit();
        qcirc::Gate1QOp gate1Q = qubit.getDefiningOp<qcirc::Gate1QOp>();
        if(!gate1Q) {
            return mlir::failure();
        }

        // Collect the single-qubit gates working on |->
        while(gate1Q.getControls().size() == 0) {
            extraGates.push_back(gate1Q);
            qubit = gate1Q.getQubit();

            gate1Q = qubit.getDefiningOp<qcirc::Gate1QOp>();
            if(!gate1Q) {
                return mlir::failure();
            }
        }

        // Check if after single-qubit gates are X gates with controls
        qcirc::Gate1QOp gateX = gate1Q;
        if(!gateX) {
            return mlir::failure();
        }

        // Check how many X gates before we see H(X(QAlloc))
        while(gateX.getGate() == qcirc::Gate1Q::X) {
            // X gates need controls
            if(gateX.getControls().size() == 0) {
                return mlir::failure();
            }
            xGates.push_back(gateX);
            qubit = gateX.getQubit();

            gateX = qubit.getDefiningOp<qcirc::Gate1QOp>();
            if(!gateX) {
                return mlir::failure();
            }
        }

        // Check for the next gates of H(X(QAlloc))
        qcirc::Gate1QOp gateH = gateX;
        deleteGates.push_back(gateH);
        if(!gateH || gateH.getGate() != qcirc::Gate1Q::H) {
            return mlir::failure();
        }

        mlir::Value qubitH = gateH.getQubit();
        qcirc::Gate1QOp gateHX = qubitH.getDefiningOp<qcirc::Gate1QOp>();
        deleteGates.push_back(gateHX);
        if(!gateHX || gateHX.getGate() != qcirc::Gate1Q::X) {
            return mlir::failure();
        }

        mlir::Value qubitHX = gateHX.getQubit();
        qcirc::QallocOp gateAlloc = qubitHX.getDefiningOp<qcirc::QallocOp>();
        deleteGates.push_back(gateAlloc);
        if(!gateAlloc) {
            return mlir::failure();
        }

        // Replace the X gates with Z gates
        for(qcirc::Gate1QOp gateX : xGates) {
            mlir::Value controlQubit = gateX.getControls()[gateX.getControls().size()-1];
            auto secondLast = gateX.getControls().end();
            secondLast -= 1;
            llvm::SmallVector<mlir::Value> tempVector(gateX.getControls().begin(), secondLast);
            // Insertion point needs to be moved before every replacement instruction
            rewriter.setInsertionPoint(gateX);
            qcirc::Gate1QOp gateZ = rewriter.create<qcirc::Gate1QOp>(gateX.getLoc(), qcirc::Gate1Q::Z, tempVector, controlQubit);

            llvm::SmallVector<mlir::Value> newValues(gateZ->result_begin(), gateZ->result_end());
            newValues.push_back(gateX.getQubit());
            rewriter.replaceOp(gateX, newValues);
        }

        // Push the IR inputs through all extra single-qubits to get rid of them
        for(qcirc::Gate1QOp gate1Q : extraGates) {
            rewriter.replaceOp(gate1Q, gate1Q.getOperands());
        }

        // Erase the qfree(H(X(qalloc))) subsequent gates
        for(auto deleteGate : deleteGates) {
            rewriter.eraseOp(deleteGate);
        }
        return mlir::success();
    }

};

// Standard Pauli group replacements (e.g., XY is Z up to a global phase)
class ReplaceTwoGatesWithOnePattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        mlir::Value qubit = gate.getQubit();
        qcirc::Gate1QOp gatePrev = qubit.getDefiningOp<qcirc::Gate1QOp>();
        if(!gatePrev) {
            return mlir::failure();
        }

        // Based on discussion, this optimization is only valid for no control bits
        if(gate.getControls().size() != 0 ||
           gatePrev.getControls().size() != 0) {
            return mlir::failure();
        }

        // Convert X(Y) -> Z or Y(X) -> Z
        if((gate.getGate() == qcirc::Gate1Q::X &&
            gatePrev.getGate() == qcirc::Gate1Q::Y) ||
           (gate.getGate() == qcirc::Gate1Q::Y &&
            gatePrev.getGate() == qcirc::Gate1Q::X)) {
            rewriter.replaceOp(gatePrev, gatePrev->getOperands());
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::Z, gate.getControls(), gate.getQubit());

            return mlir::success();
        }

        // Convert Z(Y) -> X or Y(Z) -> X
        if((gate.getGate() == qcirc::Gate1Q::Z &&
            gatePrev.getGate() == qcirc::Gate1Q::Y) ||
           (gate.getGate() == qcirc::Gate1Q::Y &&
            gatePrev.getGate() == qcirc::Gate1Q::Z)) {
            rewriter.replaceOp(gatePrev, gatePrev->getOperands());
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::X, gate.getControls(), gate.getQubit());

            return mlir::success();
        }

        // Convert X(Z) -> Y or Z(X) -> Y
        if((gate.getGate() == qcirc::Gate1Q::X &&
            gatePrev.getGate() == qcirc::Gate1Q::Z) ||
           (gate.getGate() == qcirc::Gate1Q::Z &&
            gatePrev.getGate() == qcirc::Gate1Q::X)) {
            rewriter.replaceOp(gatePrev, gatePrev->getOperands());
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::Y, gate.getControls(), gate.getQubit());

            return mlir::success();
        }

        return mlir::failure();
    }
};

// Someone who knows C++ better is free to refactor this if they know a
// better way.
struct MlirValueHash {
    auto operator()(mlir::Value v) const {
        return mlir::hash_value(v);
    }
};

// Hadamard–Pauli substitutions, e.g., HXH = Z
class ReplaceHGatesPattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        mlir::Value qubit = gate.getQubit();
        qcirc::Gate1QOp gatePrev = qubit.getDefiningOp<qcirc::Gate1QOp>();
        if(!gatePrev) {
            return mlir::failure();
        }

        mlir::Value qubitPrev = gatePrev.getQubit();
        qcirc::Gate1QOp gatePrevPrev = qubitPrev.getDefiningOp<qcirc::Gate1QOp>();
        if(!gatePrevPrev) {
            return mlir::failure();
        }

        // Convert H(X(H)) -> Z or H(Z(H)) -> X
        // Make sure both H gates are control on the same qubits and they are
        // a subset of the middle gate. I do not make any assumptions on the order
        // of the control bits over the three gates because the middle gate
        // might have more control bits.
        if(gate.getGate() == qcirc::Gate1Q::H &&
           (gatePrev.getGate() == qcirc::Gate1Q::X ||
            gatePrev.getGate() == qcirc::Gate1Q::Z) &&
           gatePrevPrev.getGate() == qcirc::Gate1Q::H) {

            // Temporary fix for specific case
            if(gate.getQubit() != gatePrev.getResult() ||
               gatePrev.getQubit() != gatePrevPrev.getResult()) {
                return mlir::failure();
            }

            // For some reason mlir::Value has no comparator automatically linked to it so I have to create my
            // own comparator so I can use std::sets with mlir::Value
            auto cmp = [](mlir::Value a, mlir::Value b) { return mlir::hash_value(a) < mlir::hash_value(b); };

            std::unordered_set<mlir::Value, MlirValueHash> gatePrevPrevCRSet(gatePrevPrev.getControlResults().begin(), gatePrevPrev.getControlResults().end());
            std::unordered_set<mlir::Value, MlirValueHash> gatePrevCSet(gatePrev.getControls().begin(), gatePrev.getControls().end());
            std::unordered_set<mlir::Value, MlirValueHash> gatePrevCRSet;
            std::unordered_set<mlir::Value, MlirValueHash> gateCSet(gate.getControls().begin(), gate.getControls().end());
            std::unordered_set<mlir::Value, MlirValueHash> intersect;

            // Make sure X gate is control on the same qubits as the H gate
            std::set_intersection(gatePrevPrevCRSet.begin(), gatePrevPrevCRSet.end(), gatePrevCSet.begin(), gatePrevCSet.end(),
                                  std::inserter(intersect, intersect.begin()), cmp);
            if(intersect != gatePrevPrevCRSet) {
                return mlir::failure();
            }

            // Make sure H gate is control on the same qubits as X gate coming
            // from the previous H gate
            auto controlResults = gatePrev.getControlResults();
            int index = 0;
            for(mlir::Value controlQubit : gatePrev.getControls()) {
                mlir::Value controlResult = controlResults[index];
                if(gatePrevPrevCRSet.count(controlQubit) > 0) {
                    gatePrevCRSet.insert(controlResult);
                }
                index += 1;
            }
            if(gatePrevCRSet != gateCSet){
                return mlir::failure();
            }

            // Push the IR inputs through H gates forward and just replace the X or Z gate with its counterpart.
            rewriter.replaceOp(gatePrevPrev, gatePrevPrev->getOperands());

            rewriter.setInsertionPoint(gatePrev);
            if(gatePrev.getGate() == qcirc::Gate1Q::X) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gatePrev, qcirc::Gate1Q::Z, gatePrev.getControls(), gatePrev.getQubit());
            }
            else {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gatePrev, qcirc::Gate1Q::X, gatePrev.getControls(), gatePrev.getQubit());
            }

            rewriter.replaceOp(gate, gate->getOperands());
            return mlir::success();
        }

        // Convert H(Y(H)) -> Y
        if(gate.getGate() == qcirc::Gate1Q::H &&
           gatePrev.getGate() == qcirc::Gate1Q::Y &&
           gatePrevPrev.getGate() == qcirc::Gate1Q::H) {
            // Based on discussion, this optimization is only valid for no control bits
            if(gate.getControls().size() != 0 ||
               gatePrev.getControls().size() != 0 ||
               gatePrevPrev.getControls().size() != 0) {
                return mlir::failure();
            }

            llvm::SmallVector<mlir::Value> replaceValues(gatePrevPrev.getControls().begin(), gatePrevPrev.getControls().end());
            replaceValues.push_back(gatePrevPrev.getQubit());
            rewriter.replaceOp(gatePrevPrev, replaceValues);

            llvm::SmallVector<mlir::Value> replaceValues2(gatePrev.getControls().begin(), gatePrev.getControls().end());
            replaceValues2.push_back(gatePrev.getQubit());
            rewriter.replaceOp(gatePrev, replaceValues2);

            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(gate, qcirc::Gate1Q::Y, gate.getControls(), gate.getQubit());
            return mlir::success();
        }

        return mlir::failure();
    }
};

// Replace self-adjoint gates, e.g., HH = identity
class ReplaceDuplicatePattern : public mlir::OpRewritePattern<qcirc::Gate1QOp> {
    using mlir::OpRewritePattern<qcirc::Gate1QOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate, mlir::PatternRewriter &rewriter) const override {
        mlir::Value qubit = gate.getQubit();
        qcirc::Gate1QOp gatePrev = qubit.getDefiningOp<qcirc::Gate1QOp>();
        if(!gatePrev) {
            return mlir::failure();
        }

        // Convert X(X(q)) -> q, Y(Y(x)) -> q, Z(Z(q)) -> q, H(H(q)) -> q
        if ((gate.getGate() == gatePrev.getGate() &&
             (gate.getGate() == qcirc::Gate1Q::X ||
              gate.getGate() == qcirc::Gate1Q::Y ||
              gate.getGate() == qcirc::Gate1Q::Z ||
              gate.getGate() == qcirc::Gate1Q::H))
            || (gatePrev.getGate() == qcirc::Gate1Q::Sdg
                && gate.getGate() == qcirc::Gate1Q::S)
            || (gatePrev.getGate() == qcirc::Gate1Q::S
                && gate.getGate() == qcirc::Gate1Q::Sdg)
            || (gatePrev.getGate() == qcirc::Gate1Q::Tdg
                && gate.getGate() == qcirc::Gate1Q::T)
            || (gatePrev.getGate() == qcirc::Gate1Q::T
                && gate.getGate() == qcirc::Gate1Q::Tdg)
            || (gatePrev.getGate() == qcirc::Gate1Q::Sxdg
                && gate.getGate() == qcirc::Gate1Q::Sx)
            || (gatePrev.getGate() == qcirc::Gate1Q::Sx
                && gate.getGate() == qcirc::Gate1Q::Sxdg)) {
            // The back-to-back same gates must have the same control bits to be optimized. In the IR,
            // the control results of the previous IR instruction (gate) must be the control
            // input of the current IR instruction (gate)
            std::unordered_set<mlir::Value, MlirValueHash> controlResultsSet(gatePrev.getControlResults().begin(), gatePrev.getControlResults().end());
            std::unordered_set<mlir::Value, MlirValueHash> controlSet(gate.getControls().begin(), gate.getControls().end());
            if(controlSet != controlResultsSet) {
                return mlir::failure();
            }

            rewriter.replaceOp(gatePrev, gatePrev->getOperands());
            rewriter.replaceOp(gate, gate->getOperands());
            return mlir::success();
        }
        return mlir::failure();
    }
};

// It's common for circuits synthesized for basis translations to have a lot
// of P(pi) gates in them. Replace those wih Z gates. (And P(pi/2) with S
// gates, and P(pi/4) with T gates, and same for their adjoints.)
class ReplacePPiWithZ :
        public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qcirc::Gate1Q1POp gate,
            mlir::PatternRewriter &rewriter) const override {
        if (gate.getGate() != qcirc::Gate1Q1P::P) {
            return mlir::failure();
        }
        mlir::FloatAttr float_attr;
        if (!mlir::matchPattern(gate.getParam(), qcirc::m_CalcConstant(&float_attr))) {
            return mlir::failure();
        }

        double theta = float_attr.getValueAsDouble();

        if (std::abs(theta - M_PI) <= ATOL
                || std::abs(theta - (-M_PI)) <= ATOL) {
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                gate, qcirc::Gate1Q::Z, gate.getControls(), gate.getQubit());
            return mlir::success();
        } else if (std::abs(theta - M_PI_2) <= ATOL) {
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                gate, qcirc::Gate1Q::S, gate.getControls(), gate.getQubit());
            return mlir::success();
        } else if (std::abs(theta - M_PI_4) <= ATOL) {
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                gate, qcirc::Gate1Q::T, gate.getControls(), gate.getQubit());
            return mlir::success();
        } else if (std::abs(theta - (-M_PI_2)) <= ATOL) {
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                gate, qcirc::Gate1Q::Sdg, gate.getControls(), gate.getQubit());
            return mlir::success();
        } else if (std::abs(theta - (-M_PI_4)) <= ATOL) {
            rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                gate, qcirc::Gate1Q::Tdg, gate.getControls(), gate.getQubit());
            return mlir::success();
        }

        return mlir::failure();
    }
};

// The lowering of superpos ops frequently generates R(pi/2)|0⟩. We should
// replace these cases with H|0⟩.
class ReplaceRyQallocWithHQalloc :
        public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qcirc::Gate1Q1POp gate,
            mlir::PatternRewriter &rewriter) const override {
        if (gate.getGate() != qcirc::Gate1Q1P::Ry) {
            return mlir::failure();
        }
        mlir::FloatAttr float_attr;
        if (!mlir::matchPattern(gate.getParam(), qcirc::m_CalcConstant(&float_attr))) {
            return mlir::failure();
        }

        double theta = float_attr.getValueAsDouble();

        qcirc::QallocOp qalloc = gate.getQubit().getDefiningOp<qcirc::QallocOp>();
        if (!qalloc) {
            return mlir::failure();
        }
        if (gate.getControls().empty()) {
            if (std::abs(theta - M_PI_2) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::H, mlir::ValueRange(), qalloc.getResult());
                return mlir::success();
            } else if (std::abs(theta - (-M_PI_2)) <= ATOL) {
                qcirc::Gate1QOp X = rewriter.create<qcirc::Gate1QOp>(
                    gate.getLoc(), qcirc::Gate1Q::X, mlir::ValueRange(), qalloc.getResult());
                qcirc::Gate1QOp H = rewriter.create<qcirc::Gate1QOp>(
                    gate.getLoc(), qcirc::Gate1Q::H, mlir::ValueRange(), X.getResult());

                rewriter.replaceOp(gate, H.getResult());
                return mlir::success();
            } else {
                return mlir::failure();
            }
        } else {
            return mlir::failure();
        }
    }
};

// Applying Rz(theta)|0⟩ is pointless because it only introduces a global
// phase. So just remove Rzs in this case.
class ReplaceRzQallocWithQalloc :
        public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qcirc::Gate1Q1POp gate,
            mlir::PatternRewriter &rewriter) const override {
        if (gate.getGate() != qcirc::Gate1Q1P::Rz) {
            return mlir::failure();
        }

        qcirc::QallocOp qalloc = gate.getQubit().getDefiningOp<qcirc::QallocOp>();
        if (!qalloc) {
            return mlir::failure();
        }

        if (gate.getControls().empty()) {
            rewriter.replaceOp(gate, qalloc.getResult());
            return mlir::success();
        } else {
            return mlir::failure();
        }
    }
};

// According to a resource estimator, Rz(theta) is more expensive than a
// Clifford gate. So replace e.g. Rz(pi) with Z and Rz(pi/2) with S.
class ReplaceRzWithZ :
        public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qcirc::Gate1Q1POp gate,
            mlir::PatternRewriter &rewriter) const override {
        if (gate.getGate() != qcirc::Gate1Q1P::Rz) {
            return mlir::failure();
        }
        mlir::FloatAttr float_attr;
        if (!mlir::matchPattern(gate.getParam(), qcirc::m_CalcConstant(&float_attr))) {
            return mlir::failure();
        }

        double phi = float_attr.getValueAsDouble();

        if (gate.getControls().empty()) {
            if (std::abs(phi - M_PI) <= ATOL || std::abs(phi - (-M_PI)) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::Z, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else if (std::abs(phi - M_PI_2) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::S, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else if (std::abs(phi - (-M_PI_2)) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::Sdg, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else if (std::abs(phi - M_PI_4) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::T, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else if (std::abs(phi - (-M_PI_4)) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::Tdg, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else {
                return mlir::failure();
            }
        }
        return mlir::failure();
    }
};

// Replace Rx(pi) with X gates (and if there are controls, deal with the
// global phase accordingly)
class ReplaceRxWithX :
        public mlir::OpRewritePattern<qcirc::Gate1Q1POp> {
    using mlir::OpRewritePattern<qcirc::Gate1Q1POp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qcirc::Gate1Q1POp gate,
            mlir::PatternRewriter &rewriter) const override {
        if (gate.getGate() != qcirc::Gate1Q1P::Rx) {
            return mlir::failure();
        }
        mlir::FloatAttr float_attr;
        if (!mlir::matchPattern(gate.getParam(), qcirc::m_CalcConstant(&float_attr))) {
            return mlir::failure();
        }

        double theta = float_attr.getValueAsDouble();

        if (gate.getControls().empty()) {
            if (std::abs(theta - M_PI) <= ATOL
                    || std::abs(theta - (-M_PI)) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::X, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else if (std::abs(theta - M_PI_2) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::Sx, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else if (std::abs(theta - (-M_PI_2)) <= ATOL) {
                rewriter.replaceOpWithNewOp<qcirc::Gate1QOp>(
                    gate, qcirc::Gate1Q::Sxdg, mlir::ValueRange(), gate.getQubit());
                return mlir::success();
            } else {
                return mlir::failure();
            }
        } else {
            mlir::Location loc = gate.getLoc();
            qcirc::Gate1QOp ccx, cs;

            if (std::abs(theta - M_PI) <= ATOL) {
                ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, gate.getControls(), gate.getQubit());
                cs = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sdg,
                    llvm::iterator_range(
                        ccx.getControlResults().begin(),
                        ccx.getControlResults().begin() +
                            (ccx.getControlResults().size()-1)),
                    ccx.getControlResults()[ccx.getControlResults().size()-1]);
            } else if (std::abs(theta - (-M_PI)) <= ATOL) {
                ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, gate.getControls(), gate.getQubit());
                cs = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::S,
                    llvm::iterator_range(
                        ccx.getControlResults().begin(),
                        ccx.getControlResults().begin() +
                            (ccx.getControlResults().size()-1)),
                    ccx.getControlResults()[ccx.getControlResults().size()-1]);
            } else if (std::abs(theta - M_PI_2) <= ATOL) {
                ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sx, gate.getControls(), gate.getQubit());
                cs = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Tdg,
                    llvm::iterator_range(
                        ccx.getControlResults().begin(),
                        ccx.getControlResults().begin() +
                            (ccx.getControlResults().size()-1)),
                    ccx.getControlResults()[ccx.getControlResults().size()-1]);
            } else if (std::abs(theta - (-M_PI_2)) <= ATOL) {
                ccx = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::Sxdg, gate.getControls(), gate.getQubit());
                cs = rewriter.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::T,
                    llvm::iterator_range(
                        ccx.getControlResults().begin(),
                        ccx.getControlResults().begin() +
                            (ccx.getControlResults().size()-1)),
                    ccx.getControlResults()[ccx.getControlResults().size()-1]);
            } else {
                return mlir::failure();
            }

            llvm::SmallVector<mlir::Value> replace_with(
                cs.getControlResults());
            replace_with.push_back(cs.getResult());
            replace_with.push_back(ccx.getResult());

            rewriter.replaceOp(gate, replace_with);
            return mlir::success();
        }
    }
};

// Remove sequences of the format:
// qfreez(G(G(...G(qalloc))))
// where G is a single-qubit gate without controls. This sequence can happen
// for trivial oracles.
class DeleteUnusedQubitsPattern :
        public mlir::OpRewritePattern<qcirc::QfreeZeroOp> {
    using mlir::OpRewritePattern<qcirc::QfreeZeroOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qcirc::QfreeZeroOp qfreez,
            mlir::PatternRewriter &rewriter) const override {
        llvm::SmallVector<mlir::Operation *> to_erase{qfreez};
        mlir::Value upstream = qfreez.getQubit();
        while (true) {
            if (qcirc::Gate1QOp gate1q
                    = upstream.getDefiningOp<qcirc::Gate1QOp>()) {
                if (!gate1q.getControls().empty()) {
                    return mlir::failure();
                }
                to_erase.push_back(gate1q);
                upstream = gate1q.getQubit();
            } else if (qcirc::Gate1Q1POp gate1q1p =
                    upstream.getDefiningOp<qcirc::Gate1Q1POp>()) {
                if (!gate1q1p.getControls().empty()) {
                    return mlir::failure();
                }
                to_erase.push_back(gate1q1p);
                upstream = gate1q1p.getQubit();
            } else if (qcirc::Gate1Q3POp gate1q3p =
                    upstream.getDefiningOp<qcirc::Gate1Q3POp>()) {
                if (!gate1q3p.getControls().empty()) {
                    return mlir::failure();
                }
                to_erase.push_back(gate1q3p);
                upstream = gate1q3p.getQubit();
            } else if (qcirc::QallocOp qalloc =
                    upstream.getDefiningOp<qcirc::QallocOp>()) {
                to_erase.push_back(qalloc);
                break;
            } else {
                return mlir::failure();
            }
        }

        for (mlir::Operation *op : to_erase) {
            rewriter.eraseOp(op);
        }
        return mlir::success();
    }
};

struct PeepholeOptimizationPass : public qcirc::PeepholeOptimizationBase<PeepholeOptimizationPass> {
    void runOnOperation() override {
        // As the P gate can be transformed to Z, we need to do this
        // replacement first.
        mlir::RewritePatternSet patternsPP(&getContext());
        patternsPP.add<
            ReplacePPiWithZ,
            ReplaceRxWithX,
            ReplaceRyQallocWithHQalloc,
            ReplaceRzQallocWithQalloc,
            ReplaceRzWithZ
        >(&getContext());
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patternsPP)))) {
            signalPassFailure();
        }

        // Separated the patterns because ZPattern and HGatesPattern was
        // causing issues at the same time.
        // Now, ReplaceZPattern or ReplaceZZeroPattern has higher priority as
        // it happens first then the other simple patterns.
        mlir::RewritePatternSet patternsZ(&getContext());
        patternsZ.add<ReplaceZPattern, ReplaceZZeroPattern>(&getContext());
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patternsZ)))) {
            signalPassFailure();
        }

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<
            ReplaceDuplicatePattern,
            ReplaceTwoGatesWithOnePattern,
            ReplaceHGatesPattern,
            DeleteUnusedQubitsPattern
        >(&getContext());
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createPeepholeOptimizationPass() {
    return std::make_unique<PeepholeOptimizationPass>();
}
