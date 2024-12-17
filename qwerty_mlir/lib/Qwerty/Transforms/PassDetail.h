//===- PassDetail.h - Qwerty Pass class details ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QWERTY_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_QWERTY_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace qwerty {

#define GEN_PASS_CLASSES
#include "Qwerty/Transforms/QwertyPasses.h.inc"

} // namespace qwerty

#endif // DIALECT_QWERTY_TRANSFORMS_PASSDETAIL_H_
