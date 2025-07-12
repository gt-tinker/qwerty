//===- QcircPasses.h - Qcirc Patterns and Passes ------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_TRANSFORMS_QCIRC_TYPES_H
#define DIALECT_INCLUDE_QCIRC_TRANSFORMS_QCIRC_TYPES_H

#include "mlir/Pass/Pass.h"
// include func::FuncOp definition for recursion to loop pass
#include "mlir/Dialect/Func/IR/FuncOps.h"
// For lowering to LLVM
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace qcirc {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createInlineAdjPass();

std::unique_ptr<mlir::Pass> createDecomposeMultiControlPass();

std::unique_ptr<mlir::Pass> createReplaceNonQIRGatesPass();

std::unique_ptr<mlir::Pass> createReplaceNonQasmGatesPass();

std::unique_ptr<mlir::Pass> createPeepholeOptimizationPass();

std::unique_ptr<mlir::Pass> createBaseProfileModulePrepPass();

std::unique_ptr<mlir::Pass> createBaseProfileFuncPrepPass();

std::unique_ptr<mlir::Pass> createQCircToQIRConversionPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "QCirc/Transforms/QCircPasses.h.inc"

} // namespace qcirc

#endif // DIALECT_INCLUDE_QCIRC_TRANSFORMS_QCIRC_TYPES_H
