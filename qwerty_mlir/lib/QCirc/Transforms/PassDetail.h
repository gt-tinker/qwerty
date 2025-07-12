//===- PassDetail.h - QCirc Pass class details ----------------*- C++ -*-===//

#ifndef DIALECT_QCIRC_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_QCIRC_TRANSFORMS_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace qcirc {

#define GEN_PASS_CLASSES
#include "QCirc/Transforms/QCircPasses.h.inc"

} // namespace qcirc

#endif // DIALECT_QCIRC_TRANSFORMS_PASSDETAIL_H_
