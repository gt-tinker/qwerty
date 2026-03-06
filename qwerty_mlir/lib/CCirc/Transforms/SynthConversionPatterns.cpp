// Must be first #include. See util.hpp.
#include "util.hpp"

#include "mlir/Transforms/DialectConversion.h"

#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"

#include "PassDetail.h"

// These are synthesis conversion patterns used by both
// CCircToXAGConversionPass and CCircToFuncArithConversionPass.

namespace {

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
        decomposeRotate(rewriter, rotl.getValue(), rotl.getAmount(),
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
        decomposeRotate(rewriter, rotr.getValue(), rotr.getAmount(),
                        rotr, ccirc::BitRotateDirection::Right);
        return mlir::success();
    }
};

struct DecomposeAdd
        : public mlir::OpConversionPattern<ccirc::AddOp> {
    using mlir::OpConversionPattern<ccirc::AddOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::AddOp add,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = add.getLoc();

        llvm::SmallVector<mlir::Value> wires_a(rewriter.create<ccirc::WireUnpackOp>(
            loc, add.getA()).getWires());
        llvm::SmallVector<mlir::Value> wires_b(rewriter.create<ccirc::WireUnpackOp>(
            loc, add.getB()).getWires());

        llvm::SmallVector<mlir::Value> wires_sum;
        ccirc::synthAdd(rewriter, loc, wires_a, wires_b, wires_sum);

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(add, wires_sum);
        return mlir::success();
    }
};

struct DecomposeModMul
        : public mlir::OpConversionPattern<ccirc::ModMulOp> {
    using mlir::OpConversionPattern<ccirc::ModMulOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::ModMulOp mod_mul,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = mod_mul.getLoc();
        uint64_t x = mod_mul.getX();
        uint64_t j = mod_mul.getJ();
        uint64_t N = mod_mul.getModN();

        // Go ahead and do the repeated squaring here, classically
        // TODO: should use llvm::APInt here in case N is very large
        uint64_t x_2j_modN = x % N;
        for (uint64_t i = 1; i <= j; i++) {
            x_2j_modN = (x_2j_modN * x_2j_modN) % N;
        }

        mlir::ValueRange y_wires = rewriter.create<ccirc::WireUnpackOp>(
            loc, mod_mul.getY()).getWires();
        llvm::SmallVector<mlir::Value> wires_y(y_wires);
        llvm::SmallVector<mlir::Value> wires_out;

        // TODO: these should be APInts already as mentioned above. That is,
        //       the AST node should also be updated to have dashu::IBigs, and
        //       the ModMulOp itself should have APInt attributes instead of i64
        //       attributes.
        llvm::APInt x_bigint(BITS_NEEDED(x_2j_modN), x_2j_modN);
        llvm::APInt modN_bigint(BITS_NEEDED(N), N);

        ccirc::synthModMul(rewriter, loc, x_bigint, modN_bigint, wires_y,
                           wires_out);

        rewriter.replaceOpWithNewOp<ccirc::WirePackOp>(mod_mul, wires_out);
        return mlir::success();
    }
};

} // namespace

// Add patterns used by both CCircToXAGConversionPass and
// CCircToFuncArithConversionPass.
void ccirc::populateSynthConversionPatterns(
        mlir::RewritePatternSet &patterns) {
    patterns.add<DecomposeRotateLeft,
                 DecomposeRotateRight,
                 DecomposeAdd,
                 DecomposeModMul>(patterns.getContext());
}
