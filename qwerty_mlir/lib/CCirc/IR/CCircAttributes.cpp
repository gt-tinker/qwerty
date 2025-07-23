//===- CCircAttributes.cpp - CCirc dialect attributes ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "CCirc/IR/CCircAttributes.h"
#include "CCirc/IR/CCircDialect.h"

#include "CCirc/IR/CCircOpsAttributes.cpp.inc"

namespace ccirc {

void CCircDialect::registerAttributes() {
    addAttributes<
#include "CCirc/IR/CCircOpsAttributes.cpp.inc"
    >();
}

} // namespace ccirc
