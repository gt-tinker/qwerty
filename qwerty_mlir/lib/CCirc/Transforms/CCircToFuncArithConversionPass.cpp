// Must be first #include. See util.hpp.
#include "util.hpp"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"
#include "CCirc/Transforms/CCircPasses.h"

#include "PassDetail.h"

// Convert ccirc.circuit ops into func.func ops containing arith ops.

namespace {

class CCircToFuncArithTypeConverter : public mlir::TypeConverter {
public:
    CCircToFuncArithTypeConverter(mlir::MLIRContext *ctx) {
        // Fallback to letting stuff through (MLIR checks the list of
        // conversions in reverse order)
        addConversion([=](mlir::Type ty) { return ty; });

        addConversion([=](ccirc::WireType wire) {
            return mlir::IntegerType::get(ctx, wire.getDim());
        });
    }
};

struct ConstantToIntConstant
        : public mlir::OpConversionPattern<ccirc::ConstantOp> {
    using mlir::OpConversionPattern<ccirc::ConstantOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::ConstantOp constant,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
            constant,
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(constant.getValue().getBitWidth()),
                constant.getValue()
            )
        );
        return mlir::success();
    }
};

struct AndToArithAnd
        : public mlir::OpConversionPattern<ccirc::AndOp> {
    using mlir::OpConversionPattern<ccirc::AndOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::AndOp and_op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
            and_op,
            adaptor.getLeft(),
            adaptor.getRight()
        );
        return mlir::success();
    }
};

struct OrToArithOr
        : public mlir::OpConversionPattern<ccirc::OrOp> {
    using mlir::OpConversionPattern<ccirc::OrOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::OrOp and_op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(
            and_op,
            adaptor.getLeft(),
            adaptor.getRight()
        );
        return mlir::success();
    }
};

struct XorToArithXor
        : public mlir::OpConversionPattern<ccirc::XorOp> {
    using mlir::OpConversionPattern<ccirc::XorOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::XorOp and_op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
            and_op,
            adaptor.getLeft(),
            adaptor.getRight()
        );
        return mlir::success();
    }
};

struct PackToArithShiftOr
        : public mlir::OpConversionPattern<ccirc::WirePackOp> {
    using mlir::OpConversionPattern<ccirc::WirePackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::WirePackOp pack,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = pack.getLoc();
        size_t combined_wire_dim = pack.getWire().getType().getDim();
        mlir::Type result_ty = rewriter.getIntegerType(combined_wire_dim);

        size_t shift_by = 0;
        mlir::Value result;

        for (auto [wire_val, int_val]
                : llvm::reverse(llvm::zip(pack.getWires(),
                                          adaptor.getWires()))) {
            size_t this_wire_dim = llvm::cast<ccirc::WireType>(
                wire_val.getType()).getDim();
            mlir::Value zext = rewriter.create<mlir::arith::ExtUIOp>(
                loc, result_ty, int_val).getResult();
            if (!result) {
                result = zext;
            } else {
                mlir::Value amt = rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getIntegerAttr(
                        rewriter.getIntegerType(combined_wire_dim), shift_by));
                mlir::Value shifted = rewriter.create<mlir::arith::ShLIOp>(
                    loc, zext, amt).getResult();
                result = rewriter.create<mlir::arith::OrIOp>(
                    loc, result, shifted).getResult();
            }
            shift_by += this_wire_dim;
        }

        rewriter.replaceOp(pack, result);
        return mlir::success();
    }
};

struct UnpackToArithShiftTrunc
        : public mlir::OpConversionPattern<ccirc::WireUnpackOp> {
    using mlir::OpConversionPattern<ccirc::WireUnpackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::WireUnpackOp unpack,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = unpack.getLoc();
        size_t in_dim = unpack.getWire().getType().getDim();
        mlir::Value wire_int_val = adaptor.getWire();
        mlir::Type i1_ty = rewriter.getI1Type();

        llvm::SmallVector<mlir::Value> results;
        for (size_t i = 0; i < in_dim; i++) {
            size_t shift_by = in_dim - 1 - i;
            mlir::Value shifted;
            if (shift_by) {
                mlir::Value amt = rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getIntegerAttr(
                        rewriter.getIntegerType(in_dim), shift_by));
                shifted = rewriter.create<mlir::arith::ShRUIOp>(
                    loc, wire_int_val, amt).getResult();
            } else {
                shifted = wire_int_val;
            }
            mlir::Value truncated = rewriter.create<mlir::arith::TruncIOp>(
                loc, i1_ty, shifted).getResult();
            results.push_back(truncated);
        }

        rewriter.replaceOp(unpack, results);
        return mlir::success();
    }
};

struct NotToXor1
        : public mlir::OpConversionPattern<ccirc::NotOp> {
    using mlir::OpConversionPattern<ccirc::NotOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::NotOp not_op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = not_op.getLoc();
        llvm::APInt all_ones_val(not_op.getResult().getType().getDim(), 0);
        all_ones_val.flipAllBits();
        mlir::Value all_ones = rewriter.create<ccirc::ConstantOp>(
            loc, all_ones_val).getResult();
        rewriter.replaceOpWithNewOp<ccirc::XorOp>(
            not_op, adaptor.getOperand(), all_ones);
        return mlir::success();
    }
};

struct ParityToXors
        : public mlir::OpConversionPattern<ccirc::ParityOp> {
    using mlir::OpConversionPattern<ccirc::ParityOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::ParityOp parity_op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = parity_op.getLoc();

        mlir::Value result;
        for (mlir::Value operand : parity_op.getOperands()) {
            if (!result) {
                result = operand;
            } else {
                result = rewriter.create<ccirc::XorOp>(
                    loc, result, operand).getResult();
            }
        }

        assert(result && "Empty parity ops are not allowed");

        rewriter.replaceOp(parity_op, result);
        return mlir::success();
    }
};

//mlir::FunctionType wire_tys_to_int_tys(mlir::Builder &builder, mlir::TypeRange tys, llvm::SmallVectorImpl<mlir::Type> &tys_out) {
//    for (mlir::Type ty : tys) {
//        size_t dim = llvm::cast<ccirc::WireType>(ty).getDim();
//        tys_out.push_back(builder.getIntegerType(dim));
//    }
//}

mlir::FunctionType convertCircuitToFunctionType(
        const mlir::TypeConverter &ty_conv, ccirc::CircuitOp circ) {
    llvm::SmallVector<mlir::Type> in_tys;
    llvm::SmallVector<mlir::Type> out_tys;

    mlir::TypeRange arg_tys = circ.bodyBlock().getArgumentTypes();
    if (mlir::failed(ty_conv.convertTypes(arg_tys, in_tys))) {
        return nullptr;
    }

    mlir::TypeRange ret_operand_tys =
        circ.bodyBlock().getTerminator()->getOperandTypes();
    if (mlir::failed(ty_conv.convertTypes(ret_operand_tys, out_tys))) {
        return nullptr;
    }

    return mlir::FunctionType::get(circ.getContext(), in_tys, out_tys);
}

struct CircuitToFunc
        : public mlir::OpConversionPattern<ccirc::CircuitOp> {
    using mlir::OpConversionPattern<ccirc::CircuitOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::CircuitOp circ,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(circ, "Need a type converter");
        }
        mlir::Location loc = circ.getLoc();
        mlir::FunctionType func_ty = convertCircuitToFunctionType(
            *type_conv, circ);
        if (!func_ty) {
            return rewriter.notifyMatchFailure(circ,
                "Could not create func type for circuit");
        }

        mlir::func::FuncOp func_op =
            rewriter.create<mlir::func::FuncOp>(
                loc, circ.getSymName(), func_ty);
        if (circ.isPrivate()) {
            func_op.setPrivate();
        }

        rewriter.inlineRegionBefore(circ.getBody(),
                                    func_op.getBody(),
                                    func_op.getBody().end());

        if (mlir::failed(rewriter.convertRegionTypes(
                &func_op.getBody(), *type_conv))) {
            return rewriter.notifyMatchFailure(circ,
                "Converting region types failed");
        }

        rewriter.eraseOp(circ);
        return mlir::success();
    }
};

struct ReturnToFuncReturn
        : public mlir::OpConversionPattern<ccirc::ReturnOp> {
    using mlir::OpConversionPattern<ccirc::ReturnOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::ReturnOp ret,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
            ret, adaptor.getOperands());
        return mlir::success();
    }
};

struct FuncPtrToFuncConst
        : public mlir::OpConversionPattern<ccirc::FuncPtrOp> {
    using mlir::OpConversionPattern<ccirc::FuncPtrOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
            ccirc::FuncPtrOp func_ptr_op,
            OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::func::ConstantOp>(
            func_ptr_op, func_ptr_op.getResult().getType(), func_ptr_op.getValue());
        return mlir::success();
    }
};

struct CCircToFuncArithConversionPass
        : public ccirc::CCircToFuncArithConversionBase<CCircToFuncArithConversionPass> {
    void runOnOperation() override {
        mlir::ConversionTarget target(getContext());
        target.addIllegalDialect<ccirc::CCircDialect>();
        target.addLegalDialect<mlir::func::FuncDialect,
                               mlir::arith::ArithDialect>();

        CCircToFuncArithTypeConverter type_converter(&getContext());
        mlir::RewritePatternSet patterns(&getContext());
        ccirc::populateSynthConversionPatterns(patterns);
        patterns.add<ConstantToIntConstant,
                     AndToArithAnd,
                     OrToArithOr,
                     XorToArithXor,
                     PackToArithShiftOr,
                     UnpackToArithShiftTrunc,
                     // These just generate more ccirc ops
                     NotToXor1,
                     ParityToXors,
                     // Final structural conversions
                     CircuitToFunc,
                     ReturnToFuncReturn,
                     // Facilitate a hack used by FileCheck tests
                     FuncPtrToFuncConst>(type_converter, &getContext());

        if (mlir::failed(mlir::applyPartialConversion(
                getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> ccirc::createCCircToFuncArithConversionPass() {
    return std::make_unique<CCircToFuncArithConversionPass>();
}
