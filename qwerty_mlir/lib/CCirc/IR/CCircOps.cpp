//===- CCircOps.cpp - CCirc dialect ops --------------------*- C++ -*-===//
//===-----------------------------------------------------------------===//

#include <unordered_set>
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "CCirc/IR/CCircOps.h"
#include "CCirc/IR/CCircDialect.h"
#include "CCirc/Transforms/CCircPasses.h"

#define GET_OP_CLASSES
#include "CCirc/IR/CCircOps.cpp.inc"

namespace {

struct DoubleNegationPattern : public mlir::OpRewritePattern<ccirc::NotOp> {
    using OpRewritePattern<ccirc::NotOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ccirc::NotOp op,
                                        mlir::PatternRewriter &rewriter) const override {
        ccirc::NotOp upstream_notop =
            op.getOperand().getDefiningOp<ccirc::NotOp>();

        if (!upstream_notop) {
            return mlir::failure();
        }

        rewriter.replaceOp(op, upstream_notop.getOperand());
        return mlir::success();
    }
};

struct SimplifyPackUnpack : public mlir::OpRewritePattern<ccirc::WireBundleUnpackOp> {
    using OpRewritePattern<ccirc::WireBundleUnpackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ccirc::WireBundleUnpackOp unpack,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value wire = unpack.getWire();
        ccirc::WireBundlePackOp pack = wire.getDefiningOp<ccirc::WireBundlePackOp>();
        if (!pack) {
            return mlir::failure();
        }
        rewriter.replaceOp(unpack, pack.getWires());
        return mlir::success();
    }
};

} // namespace

namespace ccirc {

// Based on FuncOp::parse() in QwertyOps.cpp
mlir::ParseResult CircuitOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {

    (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

    mlir::StringAttr sym_name;
    if (parser.parseSymbolName(sym_name, mlir::SymbolTable::getSymbolAttrName(),
                               result.attributes)) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::OpAsmParser::Argument> body_args;
    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::Argument arg;
                if (parser.parseArgument(arg, /*allowType=*/true, /*allowAttrs=*/false)) {
                    return mlir::failure();
                }
                body_args.push_back(arg);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    bool rev = false;
    if (!parser.parseOptionalKeyword("rev")) {
        rev = true;
    } else if (parser.parseKeyword("irrev")) {
        return {};
    }
    result.addAttribute(getReversibleAttrName(result.name),
                        mlir::BoolAttr::get(result.getContext(), rev));

    // Why print out the `attributes' keyword before the attribute dict?
    // The keyword token resolves a parsing ambiguity where the opening { of
    // the region looks like the start of an attribute dict
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
        return mlir::failure();
    }

    // Introduce the body region and parse it.
    mlir::Region *body = result.addRegion();
    if (parser.parseRegion(*body, body_args)) {
        return mlir::failure();
    }

    return mlir::success();
}

// Based on FuncOp::print() in QwertyOps.cpp
void CircuitOp::print(mlir::OpAsmPrinter &p) {
    p << ' ';
    if (getSymVisibility()) {
        p << getSymVisibility().value() << ' ';
    }
    p.printSymbolName(getSymName());

    mlir::Region &body = getBody();
    p << "(";
    for (size_t i = 0; i < body.getNumArguments(); i++) {
        if (i) {
            p << ", ";
        }
        p.printRegionArgument(body.getArgument(i));
    }
    p << ") ";

    if (getReversible()) {
        p << "rev";
    } else {
        p << "irrev";
    }

    // Why print out the `attributes' keyword before the attribute dict?
    // The keyword token resolves a parsing ambiguity where the opening { of
    // the region looks like the start of an attribute dict
    p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), {
        getReversibleAttrName(), getSymNameAttrName(), getSymVisibilityAttrName()
    });

    p << ' ';
    p.printRegion(body,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

uint64_t CircuitOp::inDim() {
    uint64_t in_dim = 0;
    for (mlir::Type arg_ty : bodyBlock().getArgumentTypes()) {
        in_dim += llvm::cast<WireBundleType>(arg_ty).getDim();
    }
    return in_dim;
}

uint64_t CircuitOp::outDim() {
    uint64_t out_dim = 0;
    auto ret_operands =
        llvm::cast<ReturnOp>(bodyBlock().getTerminator()).getOperands();
    for (mlir::Value ret_operand : ret_operands) {
        out_dim += llvm::cast<WireBundleType>(ret_operand.getType()).getDim();
    }
    return out_dim;
}

mlir::LogicalResult CircuitOp::verify() {
    if (getReversible()) {
        uint64_t in_dim = inDim();
        uint64_t out_dim = outDim();

        if (in_dim != out_dim) {
            return emitOpError("Input and output bit sizes must match in "
                               "reversible functions");
        }
    }
    return mlir::success();
}

// TODO: Create an op interface like Adjointable instead of hardcoding like
//       we do below
CircuitOp CircuitOp::buildInverseCircuit(mlir::RewriterBase &rewriter, mlir::Location loc, llvm::StringRef inv_circ_name) {
    assert(getReversible() && "Cannot take inverse of irreversible circuit");

    CircuitOp inv_circ = rewriter.create<CircuitOp>(loc, true, inv_circ_name);
    inv_circ.setPrivate();
    mlir::Block &fwd_block = bodyBlock();
    llvm::DenseMap<mlir::Value, mlir::Value> fwd_to_inv;

    for (mlir::Operation &op_ref : llvm::reverse(fwd_block)) {
        mlir::Operation *op = &op_ref;

        if (ReturnOp ret = llvm::dyn_cast<ReturnOp>(op)) {
            llvm::SmallVector<mlir::Type> inv_block_arg_tys;
            for (mlir::Value fwd_ret_operand : ret.getOperands()) {
                WireBundleType fwd_ret_operand_ty =
                    llvm::cast<WireBundleType>(fwd_ret_operand.getType());
                uint64_t dim = fwd_ret_operand_ty.getDim();
                inv_block_arg_tys.push_back(
                    rewriter.getType<WireBundleType>(dim));
            }

            llvm::SmallVector<mlir::Location> inv_block_arg_locs(
                inv_block_arg_tys.size(), loc);
            mlir::Block *inv_block = rewriter.createBlock(
                &inv_circ.getBody(), {}, inv_block_arg_tys, inv_block_arg_locs);
            assert(inv_block->getNumArguments() == ret.getOperands().size()
                   && "Wrong number of arguments for inverse blocks");

            for (auto [fwd_ret_in, inv_block_arg] :
                    llvm::zip(ret.getOperands(), inv_block->getArguments())) {
                [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                    {fwd_ret_in, inv_block_arg}).second;
                assert(inserted && "Re-encountered a return operand");
            }
        } else if (NotOp not_op = llvm::dyn_cast<NotOp>(op)) {
            assert(!inv_circ.getBody().empty()
                   && "Inverse circuit body missing");

            mlir::Value fwd_in = not_op.getOperand();
            mlir::Value fwd_out = not_op.getResult();
            assert(fwd_to_inv.contains(fwd_out)
                   && "NotOp result never encountered");

            mlir::Value inv_in = fwd_to_inv.at(fwd_out);
            mlir::Value inv_out = rewriter.create<NotOp>(
                not_op.getLoc(), inv_in).getResult();

            [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                {fwd_in, inv_out}).second;
            assert(inserted && "Re-encountered a NotOp operand");
        } else if (WireBundlePackOp pack =
                llvm::dyn_cast<WireBundlePackOp>(op)) {
            assert(!inv_circ.getBody().empty()
                   && "Inverse circuit body missing");

            mlir::ValueRange fwd_ins = pack.getWires();
            mlir::Value fwd_out = pack.getWire();
            assert(fwd_to_inv.contains(fwd_out)
                   && "NotOp result never encountered");

            mlir::Value inv_in = fwd_to_inv.at(fwd_out);
            mlir::ValueRange inv_unpacked =
                rewriter.create<WireBundleUnpackOp>(
                    pack.getLoc(), inv_in).getWires();

            uint64_t wire_idx = 0;
            for (mlir::Value fwd_in : fwd_ins) {
                WireBundleType fwd_in_ty =
                    llvm::cast<WireBundleType>(fwd_in.getType());
                uint64_t dim = fwd_in_ty.getDim();

                llvm::SmallVector<mlir::Value> wires_to_repack(
                    inv_unpacked.begin() + wire_idx,
                    inv_unpacked.begin() + (wire_idx + dim));

                mlir::Value inv_out = rewriter.create<WireBundlePackOp>(
                    loc, wires_to_repack).getWire();
                [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                    {fwd_in, inv_out}).second;
                assert(inserted && "Re-encountered a return operand");

                wire_idx += dim;
            }
        } else if (WireBundleUnpackOp unpack =
                llvm::dyn_cast<WireBundleUnpackOp>(op)) {
            assert(!inv_circ.getBody().empty()
                   && "Inverse circuit body missing");

            mlir::Value fwd_in = unpack.getWire();
            mlir::ValueRange fwd_outs = unpack.getWires();

            llvm::SmallVector<mlir::Value> inv_ins;
            for (mlir::Value fwd_out : fwd_outs) {
                assert(fwd_to_inv.contains(fwd_out)
                       && "WireBundleUnpackOp result never encountered");
                inv_ins.push_back(fwd_to_inv.at(fwd_out));
            }
            mlir::Value inv_out = rewriter.create<WireBundlePackOp>(
                loc, inv_ins).getWire();
            [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                {fwd_in, inv_out}).second;
            assert(inserted
                   && "Re-encountered a WireBundleUnpackOp operand");
        } else {
            op->dump();
            assert(0 && "Missing handling for op in buildInverseCircuit()");
        }
    }

    llvm::SmallVector<mlir::Value> ret_operands;
    for (mlir::BlockArgument fwd_block_arg : fwd_block.getArguments()) {
        assert(fwd_to_inv.contains(fwd_block_arg)
               && "fwd block arg never encountered");
        ret_operands.push_back(fwd_to_inv.at(fwd_block_arg));
    }
    rewriter.create<ReturnOp>(loc, ret_operands);

    return inv_circ;
}

mlir::LogicalResult ConstantOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        ConstantOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    inferredReturnTypes.push_back(
        WireBundleType::get(ctx, adaptor.getValue().getBitWidth()));
    return mlir::success();
}

#define BINARY_OP_VERIFY_AND_INFER(name) \
    mlir::LogicalResult name##Op::verify() { \
        if (getLeft().getType() != getRight().getType()) { \
            return emitOpError("Left and right wire sizes do not match"); \
        } \
        if (getLeft().getType() != getResult().getType()) { \
            return emitOpError("Input and output wire sizes do not match"); \
        } \
        return mlir::success(); \
    } \
    mlir::LogicalResult name##Op::inferReturnTypes( \
            mlir::MLIRContext *ctx, \
            std::optional<mlir::Location> loc, \
            name##Op::Adaptor adaptor, \
            llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) { \
        inferredReturnTypes.push_back(adaptor.getLeft().getType()); \
        return mlir::success(); \
    }

BINARY_OP_VERIFY_AND_INFER(And)
BINARY_OP_VERIFY_AND_INFER(Or)
BINARY_OP_VERIFY_AND_INFER(Xor)

#define UNARY_OP_VERIFY_AND_INFER(name) \
    mlir::LogicalResult name##Op::verify() { \
        if (getOperand().getType() != getResult().getType()) { \
            return emitOpError("Input and output wire sizes do not match"); \
        } \
        return mlir::success(); \
    } \
    mlir::LogicalResult name##Op::inferReturnTypes( \
            mlir::MLIRContext *ctx, \
            std::optional<mlir::Location> loc, \
            name##Op::Adaptor adaptor, \
            llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) { \
        inferredReturnTypes.push_back(adaptor.getOperand().getType()); \
        return mlir::success(); \
    }

UNARY_OP_VERIFY_AND_INFER(Not)

void NotOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        mlir::MLIRContext *context) {
    results.add<DoubleNegationPattern>(context);
}

mlir::LogicalResult WireBundlePackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        WireBundlePackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    size_t combined_dim = 0;
    for (mlir::Value wire : adaptor.getWires()) {
        WireBundleType wire_ty = llvm::dyn_cast<WireBundleType>(wire.getType());
        if (!wire_ty) {
            return mlir::failure();
        }
        combined_dim += wire_ty.getDim();
    }
    WireBundleType ret_ty = WireBundleType::get(ctx, combined_dim);
    inferredReturnTypes.insert(inferredReturnTypes.end(), ret_ty);
    return mlir::success();
}

mlir::LogicalResult WireBundleUnpackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        WireBundleUnpackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    WireBundleType in_ty = llvm::dyn_cast<WireBundleType>(adaptor.getWire().getType());
    if (!in_ty) {
        return mlir::failure();
    }
    size_t in_dim = in_ty.getDim();
    inferredReturnTypes.append(in_dim, WireBundleType::get(ctx, 1));
    return mlir::success();
}

void WireBundleUnpackOp::getCanonicalizationPatterns(
        mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<SimplifyPackUnpack>(context);
}

} // namespace ccirc
