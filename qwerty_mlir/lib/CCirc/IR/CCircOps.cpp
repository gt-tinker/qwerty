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

} // namespace ccirc
