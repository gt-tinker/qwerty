//===- PassDetail.h - CCirc Pass class details ----------------*- C++ -*-===//

#ifndef DIALECT_CCIRC_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_CCIRC_TRANSFORMS_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace ccirc {

#define GEN_PASS_CLASSES
#include "CCirc/Transforms/CCircPasses.h.inc"

} // namespace ccirc

#endif // DIALECT_CCIRC_TRANSFORMS_PASSDETAIL_H_
