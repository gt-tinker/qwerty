//===- QCircTypes.cpp - QCirc dialect types -----------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircTypes.h"

#define GET_TYPEDEF_CLASSES
#include "QCirc/IR/QCircOpsTypes.cpp.inc"

namespace qcirc {

void QCircDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "QCirc/IR/QCircOpsTypes.cpp.inc"
    >();
}

} // namespace qcirc
