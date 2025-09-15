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

#define GET_OP_CLASSES
#include "CCirc/IR/CCircOps.cpp.inc"

namespace {

// Extended Euclidean Algorithm from
// https://cp-algorithms.com/algebra/extended-euclid-algorithm.html
uint64_t extended_euclidean(uint64_t a, uint64_t b, uint64_t &x, uint64_t &y) {
    x = 1, y = 0;
    uint64_t x1 = 0, y1 = 1, a1 = a, b1 = b;
    while (b1) {
        uint64_t q = a1 / b1;
        std::tie(x, x1) = std::make_tuple(x1, x - q * x1);
        std::tie(y, y1) = std::make_tuple(y1, y - q * y1);
        std::tie(a1, b1) = std::make_tuple(b1, a1 - q * b1);
    }
    return a1;
}

// Modular inverse from
// https://cp-algorithms.com/algebra/module-inverse.html
uint64_t modular_inverse(uint64_t a, uint64_t m) {
    uint64_t x, y;
    uint64_t g = extended_euclidean(a, m, x, y);
    assert(g == 1 && "Modular inverse does not exist");
    return (x % m + m) % m;
}

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

struct SimplifyPackUnpack : public mlir::OpRewritePattern<ccirc::WireUnpackOp> {
    using OpRewritePattern<ccirc::WireUnpackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ccirc::WireUnpackOp unpack,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value wire = unpack.getWire();
        ccirc::WirePackOp pack = wire.getDefiningOp<ccirc::WirePackOp>();
        if (!pack) {
            return mlir::failure();
        }
        rewriter.replaceOp(unpack, pack.getWires());
        return mlir::success();
    }
};

struct SimplifyTrivialPack : public mlir::OpRewritePattern<ccirc::WirePackOp> {
    using OpRewritePattern<ccirc::WirePackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ccirc::WirePackOp pack,
                                        mlir::PatternRewriter &rewriter) const override {
        if (pack.getWires().size() != 1) {
            return mlir::failure();
        }

        rewriter.replaceOp(pack, pack.getWires());
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
        in_dim += llvm::cast<WireType>(arg_ty).getDim();
    }
    return in_dim;
}

uint64_t CircuitOp::outDim() {
    uint64_t out_dim = 0;
    auto ret_operands =
        llvm::cast<ReturnOp>(bodyBlock().getTerminator()).getOperands();
    for (mlir::Value ret_operand : ret_operands) {
        out_dim += llvm::cast<WireType>(ret_operand.getType()).getDim();
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
                WireType fwd_ret_operand_ty =
                    llvm::cast<WireType>(fwd_ret_operand.getType());
                uint64_t dim = fwd_ret_operand_ty.getDim();
                inv_block_arg_tys.push_back(
                    rewriter.getType<WireType>(dim));
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
        } else if (ModMulOp mod_mul = llvm::dyn_cast<ModMulOp>(op)) {
            assert(!inv_circ.getBody().empty()
                   && "Inverse circuit body missing");

            mlir::Value fwd_in = mod_mul.getY();
            mlir::Value fwd_out = mod_mul.getProduct();
            assert(fwd_to_inv.contains(fwd_out)
                   && "NotOp result never encountered");

            mlir::Value inv_in = fwd_to_inv.at(fwd_out);
            uint64_t x_inv = modular_inverse(mod_mul.getX(),
                                             mod_mul.getModN());
            mlir::Value inv_out = rewriter.create<ModMulOp>(
                mod_mul.getLoc(), x_inv, mod_mul.getJ(),
                mod_mul.getModN(), inv_in).getResult();

            [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                {fwd_in, inv_out}).second;
            assert(inserted && "Re-encountered a ModMulOp operand");
        } else if (WirePackOp pack =
                llvm::dyn_cast<WirePackOp>(op)) {
            assert(!inv_circ.getBody().empty()
                   && "Inverse circuit body missing");

            mlir::ValueRange fwd_ins = pack.getWires();
            mlir::Value fwd_out = pack.getWire();
            assert(fwd_to_inv.contains(fwd_out)
                   && "NotOp result never encountered");

            mlir::Value inv_in = fwd_to_inv.at(fwd_out);
            mlir::ValueRange inv_unpacked =
                rewriter.create<WireUnpackOp>(
                    pack.getLoc(), inv_in).getWires();

            uint64_t wire_idx = 0;
            for (mlir::Value fwd_in : fwd_ins) {
                WireType fwd_in_ty =
                    llvm::cast<WireType>(fwd_in.getType());
                uint64_t dim = fwd_in_ty.getDim();

                llvm::SmallVector<mlir::Value> wires_to_repack(
                    inv_unpacked.begin() + wire_idx,
                    inv_unpacked.begin() + (wire_idx + dim));

                mlir::Value inv_out = rewriter.create<WirePackOp>(
                    loc, wires_to_repack).getWire();
                [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                    {fwd_in, inv_out}).second;
                assert(inserted && "Re-encountered a return operand");

                wire_idx += dim;
            }
        } else if (WireUnpackOp unpack =
                llvm::dyn_cast<WireUnpackOp>(op)) {
            assert(!inv_circ.getBody().empty()
                   && "Inverse circuit body missing");

            mlir::Value fwd_in = unpack.getWire();
            mlir::ValueRange fwd_outs = unpack.getWires();

            llvm::SmallVector<mlir::Value> inv_ins;
            for (mlir::Value fwd_out : fwd_outs) {
                assert(fwd_to_inv.contains(fwd_out)
                       && "WireUnpackOp result never encountered");
                inv_ins.push_back(fwd_to_inv.at(fwd_out));
            }
            mlir::Value inv_out = rewriter.create<WirePackOp>(
                loc, inv_ins).getWire();
            [[maybe_unused]] bool inserted = fwd_to_inv.insert(
                {fwd_in, inv_out}).second;
            assert(inserted
                   && "Re-encountered a WireUnpackOp operand");
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

mlir::LogicalResult ConstantOp::verify() {
    if (getValue().getBitWidth() != getResult().getType().getDim()) {
        return emitOpError("dim of result type does not size of value");
    }

    return mlir::success();
}

mlir::LogicalResult ConstantOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        ConstantOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    inferredReturnTypes.push_back(
        WireType::get(ctx, adaptor.getValue().getBitWidth()));
    return mlir::success();
}

mlir::OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
    return getValueAttr();
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

mlir::LogicalResult ParityOp::verify() {
    if (getInputs().empty()) {
        return emitOpError("Parity needs at least one operand");
    }
    mlir::Type first_ty = getInputs()[0].getType();

    for (mlir::Type operand_ty : getInputs().getTypes()) {
        if (operand_ty != first_ty) {
            return emitOpError("Operand wire sizes do not match");
        }
    }

    return mlir::success();
}
mlir::LogicalResult ParityOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        ParityOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    if (adaptor.getInputs().empty()) {
        return mlir::failure();
    }
    inferredReturnTypes.push_back(adaptor.getInputs()[0].getType());
    return mlir::success();
}

mlir::LogicalResult ModMulOp::verify() {
    if (getY().getType() != getProduct().getType()) {
        return emitOpError("Input and output wire sizes do not match");
    }
    return mlir::success();
}

mlir::LogicalResult ModMulOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        ModMulOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    inferredReturnTypes.push_back(adaptor.getY().getType());
    return mlir::success();
}

mlir::LogicalResult WirePackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        WirePackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    size_t combined_dim = 0;
    for (mlir::Value wire : adaptor.getWires()) {
        WireType wire_ty = llvm::dyn_cast<WireType>(wire.getType());
        if (!wire_ty) {
            return mlir::failure();
        }
        combined_dim += wire_ty.getDim();
    }
    WireType ret_ty = WireType::get(ctx, combined_dim);
    inferredReturnTypes.insert(inferredReturnTypes.end(), ret_ty);
    return mlir::success();
}

void WirePackOp::getCanonicalizationPatterns(
        mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<SimplifyTrivialPack>(context);
}

mlir::LogicalResult WireUnpackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        WireUnpackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    WireType in_ty = llvm::dyn_cast<WireType>(adaptor.getWire().getType());
    if (!in_ty) {
        return mlir::failure();
    }
    size_t in_dim = in_ty.getDim();
    inferredReturnTypes.append(in_dim, WireType::get(ctx, 1));
    return mlir::success();
}

void WireUnpackOp::getCanonicalizationPatterns(
        mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    results.add<SimplifyPackUnpack>(context);
}

} // namespace ccirc
