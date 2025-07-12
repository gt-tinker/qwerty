//===- QwertyPasses.h - Qwerty Patterns and Passes ------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on Qwerty operations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_TRANSFORMS_QWERTY_PASSES_H
#define DIALECT_INCLUDE_QWERTY_TRANSFORMS_QWERTY_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "QCirc/IR/QCircDialect.h"

namespace qwerty {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLiftLambdasPass();
std::unique_ptr<mlir::Pass> createOnlyPredOnesPass();
std::unique_ptr<mlir::Pass> createQwertyToQCircConversionPass();
std::unique_ptr<mlir::Pass> createInlinePredPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "Qwerty/Transforms/QwertyPasses.h.inc"

} // namespace qwerty

#endif // DIALECT_INCLUDE_QWERTY_TRANSFORMS_QWERTY_PASSES_H
