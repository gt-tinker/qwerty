//===- QwertyTypes.cpp - Qwerty dialect types -----------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Qwerty/IR/QwertyOpsTypes.cpp.inc"

namespace qwerty {

mlir::LogicalResult BitBundleType::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        uint64_t dim) {
    if (!dim) {
        return emitError() << "BitBundle cannot be empty";
    }
    return mlir::success();
}

uint64_t QBundleType::getNumQubits() const {
    return getDim();
}

mlir::LogicalResult QBundleType::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        uint64_t dim) {
    if (!dim) {
        return emitError() << "QBundle cannot be empty";
    }
    return mlir::success();
}

mlir::LogicalResult FunctionType::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        mlir::FunctionType inner_func_type,
        bool rev) {
    if (rev) {
        if (inner_func_type.getNumInputs() != 1) {
            return emitError() << "Reversible functions must have 1 input "
                               << "(a qbundle). Instead got "
                               << inner_func_type.getNumInputs();
        }
        if (inner_func_type.getNumResults() != 1) {
            return emitError() << "Reversible functions must have 1 output: "
                               << "(a qbundle). Instead got "
                               << inner_func_type.getNumResults();
        }
        QBundleType qbundle_in =
            llvm::dyn_cast<QBundleType>(
                inner_func_type.getInputs()[0]);
        if (!qbundle_in) {
            return emitError() << "Reversible functions must take a qbundle "
                               << "as an argument";
        }

        QBundleType qbundle_out =
            llvm::dyn_cast<QBundleType>(
                inner_func_type.getResults()[0]);
        if (!qbundle_out) {
            return emitError() << "Reversible functions must return a qbundle "
                               << "as an argument";
        }

        if (qbundle_in.getDim() != qbundle_out.getDim()) {
            return emitError() << "Expected dimensions of qbundles to match, "
                               << "but " << qbundle_in.getDim() << " != "
                               << qbundle_out.getDim();
        }
    }
    return mlir::success();
}

mlir::Type FunctionType::parse(mlir::AsmParser &parser) {
    llvm::SmallVector<mlir::Type> inputs;
    if (parser.parseLParen()
            || parser.parseTypeList(inputs)
            || parser.parseRParen()) {
        return {};
    }

    bool rev = false;
    if (!parser.parseOptionalKeyword("rev")) {
        rev = true;
    } else if (parser.parseKeyword("irrev")) {
        return {};
    }

    llvm::SmallVector<mlir::Type> results;
    if (parser.parseOptionalArrowTypeList(results)) {
        return {};
    }

    return FunctionType::get(parser.getContext(),
                             mlir::FunctionType::get(parser.getContext(),
                                                     inputs,
                                                     results),
                             rev);
}

void FunctionType::print(mlir::AsmPrinter &p) const {
    p << '(';
    p << getFunctionType().getInputs();
    p << ") ";
    if (getReversible()) {
        p << "rev";
    } else {
        p << "irrev";
    }

    // Modified version of p.printOptionalArrowTypeList() inlined here because
    // I don't want a space before the arrow
    mlir::TypeRange results = getFunctionType().getResults();
    if (!results.empty()) {
        p << "-> ";

        bool wrapped = !llvm::hasSingleElement(results) ||
                       llvm::isa<FunctionType>(*results.begin());
        if (wrapped) {
            p << '(';
        }
        llvm::interleaveComma(results, p);
        if (wrapped) {
            p << ')';
        }
    }
}

void QwertyDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Qwerty/IR/QwertyOpsTypes.cpp.inc"
    >();
}

} // namespace qwerty
