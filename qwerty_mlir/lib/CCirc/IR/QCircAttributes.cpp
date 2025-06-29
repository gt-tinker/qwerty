//===- QCircAttributes.cpp - QCirc dialect attributes ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "QCirc/IR/QCircAttributes.h"
#include "QCirc/IR/QCircDialect.h"

#include "QCirc/IR/QCircOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "QCirc/IR/QCircOpsAttributes.cpp.inc"

namespace qcirc {

void FuncSpecAttr::print(mlir::AsmPrinter &printer) const {
    printer << '(';
    if (getAdjoint()) {
        printer << "adj,";
    } else {
        printer << "fwd,";
    }
    printer << getNumControls();
    printer << ',';
    printer.printAttributeWithoutType(getSymbol());
    printer << ',';
    printer.printType(getFunctionType());
    printer << ')';
}

mlir::Attribute FuncSpecAttr::parse(mlir::AsmParser &parser, mlir::Type odsType) {
    if (parser.parseLParen()) {
        return {};
    }
    bool adj;
    if (!parser.parseOptionalKeyword("adj")) {
        adj = true;
    } else if (!parser.parseOptionalKeyword("fwd")) {
        adj = false;
    } else {
        return {};
    }
    if (parser.parseComma()) {
        return {};
    }
    uint64_t n_controls = 0;
    if (parser.parseInteger(n_controls)) {
        return {};
    }
    if (parser.parseComma()) {
        return {};
    }
    mlir::StringAttr symbol;
    if (parser.parseSymbolName(symbol)) {
        return {};
    }
    if (parser.parseComma()) {
        return {};
    }
    mlir::FunctionType func_ty;
    if (parser.parseType(func_ty)) {
        return {};
    }
    if (parser.parseRParen()) {
        return {};
    }

    return FuncSpecAttr::get(parser.getContext(),
                             adj,
                             n_controls,
                             mlir::FlatSymbolRefAttr::get(symbol),
                             func_ty);
}

void QCircDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "QCirc/IR/QCircOpsAttributes.cpp.inc"
    >();
}

} // namespace qcirc
