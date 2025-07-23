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

void CCircDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "CCirc/IR/CCircOpsTypes.cpp.inc"
    >();
}

} // namespace ccirc
