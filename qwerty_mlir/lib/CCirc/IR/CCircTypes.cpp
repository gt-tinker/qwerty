//===- CCircTypes.cpp - CCirc dialect types -----------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#include "CCirc/IR/CCircDialect.h"
#include "CCirc/IR/CCircTypes.h"

#define GET_TYPEDEF_CLASSES
#include "CCirc/IR/CCircOpsTypes.cpp.inc"

namespace ccirc {

mlir::LogicalResult WireBundleType::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        uint64_t dim) {
    if (!dim) {
        return emitError() << "WireBundle cannot be empty";
    }
    return mlir::success();
}

void CCircDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "CCirc/IR/CCircOpsTypes.cpp.inc"
    >();
}

} // namespace ccirc
