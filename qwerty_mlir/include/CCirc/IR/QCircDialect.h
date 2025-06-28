//===- QCircDialect.h - QCirc dialect -----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_DIALECT_H
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_DIALECT_H

#include "mlir/IR/Dialect.h"

#include "QCirc/IR/QCircOpsDialect.h.inc"

namespace qcirc {

// Needed for the unfortunate LLVMConstantArrayOp (see QCircOps.td)
void registerQCircDialectTranslation(mlir::DialectRegistry &registry);

} // namespace qcirc

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_DIALECT_H
