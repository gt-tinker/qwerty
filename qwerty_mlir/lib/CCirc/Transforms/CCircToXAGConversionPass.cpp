#include "mlir/Transforms/DialectConversion.h"

#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"
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

void decomposeRotate(
        mlir::ConversionPatternRewriter &rewriter,
        mlir::Value value,
        mlir::Value amt,
        mlir::Operation *op,
        ccirc::BitRotateDirection dir) {
    mlir::Location loc = op->getLoc();
    mlir::ValueRange value_wires = rewriter.create<ccirc::WireUnpackOp>(
        loc, value).getWires();
    llvm::SmallVector<mlir::Value> wires_n(value_wires);

    mlir::ValueRange amount_wires = rewriter.create<ccirc::WireUnpackOp>(
        loc, amt).getWires();
    llvm::SmallVector<mlir::Value> wires_k(amount_wires);

    llvm::SmallVector<mlir::Value> wires_out;
    ccirc::synthBitRotate(rewriter, loc, dir, wires_n, wires_k, wires_out);
    rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(op, wires_out);
}

struct DecomposeRotateLeft
        : public mlir::OpConversionPattern<ccirc::RotateLeftOp> {
    using mlir::OpConversionPattern<ccirc::RotateLeftOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::RotateLeftOp rotl,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        decomposeRotate(rewriter, adaptor.getValue(), adaptor.getAmount(),
                        rotl, ccirc::BitRotateDirection::Left);
        return mlir::success();
    }
};

struct DecomposeRotateRight
        : public mlir::OpConversionPattern<ccirc::RotateRightOp> {
    using mlir::OpConversionPattern<ccirc::RotateRightOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::RotateRightOp rotr,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        decomposeRotate(rewriter, adaptor.getValue(), adaptor.getAmount(),
                        rotr, ccirc::BitRotateDirection::Right);
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
                if (op->getNumResults() == 1) {
                    return false;
                }
                if (ccirc::WireType res_ty = llvm::dyn_cast<ccirc::WireType>(op->getResult(0).getType())) {
                    return res_ty.getDim() == 1;
                } else {
                    return false;
                }
            });

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<SplitMultiBitConstant,
                     DecomposeRotateLeft,
                     DecomposeRotateRight>(&getContext());

        if (mlir::failed(mlir::applyFullConversion(circ, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> ccirc::createCCircToXAGConversionPass() {
    return std::make_unique<CCircToXAGConversionPass>();
}
