//===- CCircPasses.h - Qcirc Patterns and Passes ------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on classical circuits.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H
#define DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H

#include "mlir/Pass/Pass.h"

namespace ccirc {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createCCircToXAGConversionPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "CCirc/Transforms/CCircPasses.h.inc"

} // namespace ccirc

#endif // DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H
