// Must be first #include. See util.hpp.
#include "util.hpp"

#include "mlir/Transforms/DialectConversion.h"

#include "CCirc/IR/CCircOps.h"
#include "CCirc/Transforms/CCircPasses.h"

#include "PassDetail.h"

// This pass makes a ccirc.circuit op meet the following requirements:
// 1. Semantics are identical to the original circuit
// 2. Every child ops is either structural (ccirc.return, ccirc.wirepack,
//    ccirc.wireunpack)
// 3. ...or every child op is one of the following logical ops with 1-bit
//    operands and 1-bit results:
//    a) ccirc.not
//    b) ccirc.and
//    c) ccirc.parity
//    d) ccirc.constant
// Other logic ops (e.g., ccirc.or) are initially decomposed into intermediate
// multi-bit versions of the above ops, which are then decomposed into
// structural ops and logical ops (a)-(d) above.
// Running the canonicalizer after this pass will remove redundant
// packing/unpacking, deduplicate constant ops, simplify idioms such as
// `x XOR 0`, merge parity ops, and propagate not ops through parity ops,
// producing a high-level XAG [1]. That is, a digital logic circuit containing
// only 1-bit ANDs, 1-bit parity ops whose results may be negated, and an
// optional final NOT on outputs [1].
//
// [1]: https://doi.org/10.23919/DATE51398.2021.9474163

namespace {

struct SplitMultiBitConstant
        : public mlir::OpConversionPattern<ccirc::ConstantOp> {
    using mlir::OpConversionPattern<ccirc::ConstantOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::ConstantOp constant,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (constant.getResult().getType().getDim() <= 1) {
            // Nothing to do, already single-bit
            return mlir::failure();
        }

        mlir::Location loc = constant.getLoc();
        mlir::Type i1_ty = rewriter.getIntegerType(1);
        mlir::IntegerAttr i1_zero = rewriter.getIntegerAttr(i1_ty, 0);
        mlir::IntegerAttr i1_one = rewriter.getIntegerAttr(i1_ty, 1);

        llvm::APInt bits = constant.getValue();
        llvm::SmallVector<mlir::Value> const_wires;

        for (size_t i = 0; i < bits.getBitWidth(); i++) {
            mlir::IntegerAttr bit = bits[bits.getBitWidth()-1-i]
                                    ? i1_one
                                    : i1_zero;
            mlir::Value const_wire = rewriter.create<ccirc::ConstantOp>(
                loc, bit).getResult();
            const_wires.push_back(const_wire);
        }

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(constant, const_wires);
        return mlir::success();
    }
};

// Split multi-bit AND into element-wise 1-bit ANDs.
struct SplitMultiBitAnd
        : public mlir::OpConversionPattern<ccirc::AndOp> {
    using mlir::OpConversionPattern<ccirc::AndOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::AndOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getResult().getType().getDim() <= 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        mlir::ValueRange left_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getLeft()).getWires();
        mlir::ValueRange right_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getRight()).getWires();

        llvm::SmallVector<mlir::Value> result_wires;
        for (auto [l, r] : llvm::zip(left_wires, right_wires)) {
            result_wires.push_back(
                rewriter.create<ccirc::AndOp>(loc, l, r).getResult());
        }

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(op, result_wires);
        return mlir::success();
    }
};

// Split multi-bit NOT into element-wise 1-bit NOTs.
struct SplitMultiBitNot
        : public mlir::OpConversionPattern<ccirc::NotOp> {
    using mlir::OpConversionPattern<ccirc::NotOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::NotOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getResult().getType().getDim() <= 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        mlir::ValueRange wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getOperand()).getWires();

        llvm::SmallVector<mlir::Value> result_wires;
        for (mlir::Value w : wires) {
            result_wires.push_back(
                rewriter.create<ccirc::NotOp>(loc, w).getResult());
        }

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(op, result_wires);
        return mlir::success();
    }
};

// Split multi-bit PARITY into per-bit 1-bit parities.
// parity(a[N], b[N], ...) becomes wirepack of parity(a[i], b[i], ...) for each i.
struct SplitMultiBitParity
        : public mlir::OpConversionPattern<ccirc::ParityOp> {
    using mlir::OpConversionPattern<ccirc::ParityOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::ParityOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getType().getDim() <= 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        uint64_t dim = op.getType().getDim();

        llvm::SmallVector<llvm::SmallVector<mlir::Value>> unpacked;
        for (mlir::Value operand : op.getOperands()) {
            mlir::ValueRange wires = rewriter.create<ccirc::WireUnpackOp>(
                loc, operand).getWires();
            unpacked.push_back(llvm::SmallVector<mlir::Value>(wires));
        }

        llvm::SmallVector<mlir::Value> result_wires;
        for (uint64_t i = 0; i < dim; i++) {
            llvm::SmallVector<mlir::Value> per_bit;
            for (auto &unp : unpacked) {
                per_bit.push_back(unp[i]);
            }
            result_wires.push_back(
                rewriter.create<ccirc::ParityOp>(loc, per_bit).getResult());
        }

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(op, result_wires);
        return mlir::success();
    }
};

// Split multi-bit OR into element-wise 1-bit ORs.
// Single-bit ORs are then handled by DecomposeOrToDeMorgan.
struct SplitMultiBitOr
        : public mlir::OpConversionPattern<ccirc::OrOp> {
    using mlir::OpConversionPattern<ccirc::OrOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::OrOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getResult().getType().getDim() <= 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        mlir::ValueRange left_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getLeft()).getWires();
        mlir::ValueRange right_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getRight()).getWires();

        llvm::SmallVector<mlir::Value> result_wires;
        for (auto [l, r] : llvm::zip(left_wires, right_wires)) {
            result_wires.push_back(
                rewriter.create<ccirc::OrOp>(loc, l, r).getResult());
        }

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(op, result_wires);
        return mlir::success();
    }
};

// Split multi-bit XOR into element-wise 1-bit XORs.
// Single-bit XORs are then handled by DecomposeXorToParity.
struct SplitMultiBitXor
        : public mlir::OpConversionPattern<ccirc::XorOp> {
    using mlir::OpConversionPattern<ccirc::XorOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::XorOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getResult().getType().getDim() <= 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        mlir::ValueRange left_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getLeft()).getWires();
        mlir::ValueRange right_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, op.getRight()).getWires();

        llvm::SmallVector<mlir::Value> result_wires;
        for (auto [l, r] : llvm::zip(left_wires, right_wires)) {
            result_wires.push_back(
                rewriter.create<ccirc::XorOp>(loc, l, r).getResult());
        }

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(op, result_wires);
        return mlir::success();
    }
};

// Decompose single-bit OR via De Morgan's law:
// or(a, b) = not(and(not(a), not(b)))
struct DecomposeOrToDeMorgan
        : public mlir::OpConversionPattern<ccirc::OrOp> {
    using mlir::OpConversionPattern<ccirc::OrOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::OrOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getResult().getType().getDim() != 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        mlir::Value not_left = rewriter.create<ccirc::NotOp>(
            loc, op.getLeft()).getResult();
        mlir::Value not_right = rewriter.create<ccirc::NotOp>(
            loc, op.getRight()).getResult();
        mlir::Value and_result = rewriter.create<ccirc::AndOp>(
            loc, not_left, not_right).getResult();
        rewriter.replaceOpWithNewOp<ccirc::NotOp>(op, and_result);
        return mlir::success();
    }
};

// Decompose single-bit XOR into parity:
// xor(a, b) = parity(a, b)
struct DecomposeXorToParity
        : public mlir::OpConversionPattern<ccirc::XorOp> {
    using mlir::OpConversionPattern<ccirc::XorOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::XorOp op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        if (op.getResult().getType().getDim() != 1) {
            return mlir::failure();
        }

        mlir::Location loc = op.getLoc();
        llvm::SmallVector<mlir::Value> operands = {op.getLeft(), op.getRight()};
        rewriter.replaceOpWithNewOp<ccirc::ParityOp>(
            op, op.getResult().getType(), operands);
        return mlir::success();
    }
};

struct CCircToXAGConversionPass
        : public ccirc::CCircToXAGConversionBase<CCircToXAGConversionPass> {
    void runOnOperation() override {
        ccirc::CircuitOp circ = getOperation();

        mlir::ConversionTarget target(getContext());
        target.addLegalOp<ccirc::CircuitOp,
                          ccirc::ReturnOp,
                          // Fingers crossed canonicalization will get rid of
                          // these guys
                          ccirc::WirePackOp,
                          ccirc::WireUnpackOp>();
        // A XAG should not have any multi-bit bitwise operations
        target.addDynamicallyLegalOp<ccirc::AndOp,
                                     ccirc::NotOp,
                                     ccirc::ParityOp,
                                     ccirc::ConstantOp>(
            [](mlir::Operation *op) {
                if (op->getNumResults() != 1) {
                    return false;
                }
                if (ccirc::WireType res_ty = llvm::dyn_cast<ccirc::WireType>(op->getResult(0).getType())) {
                    return res_ty.getDim() == 1;
                } else {
                    return false;
                }
            });

        mlir::RewritePatternSet patterns(&getContext());
        ccirc::populateSynthConversionPatterns(patterns);
        patterns.add<SplitMultiBitConstant,
                     SplitMultiBitAnd,
                     SplitMultiBitNot,
                     SplitMultiBitParity,
                     SplitMultiBitOr,
                     SplitMultiBitXor,
                     DecomposeOrToDeMorgan,
                     DecomposeXorToParity>(&getContext());

        if (mlir::failed(mlir::applyFullConversion(circ, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> ccirc::createCCircToXAGConversionPass() {
    return std::make_unique<CCircToXAGConversionPass>();
}
