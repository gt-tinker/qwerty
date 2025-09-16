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
