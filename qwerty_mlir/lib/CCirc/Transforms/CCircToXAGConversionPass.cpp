#include "mlir/Transforms/DialectConversion.h"

#include "CCirc/IR/CCircOps.h"
#include "CCirc/Transforms/CCircPasses.h"

#include "PassDetail.h"

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
        patterns.add<SplitMultiBitConstant>(&getContext());

        if (mlir::failed(mlir::applyFullConversion(circ, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> ccirc::createCCircToXAGConversionPass() {
    return std::make_unique<CCircToXAGConversionPass>();
}
