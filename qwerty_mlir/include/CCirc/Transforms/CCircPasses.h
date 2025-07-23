//===- CcircPasses.h - Ccirc Patterns and Passes ------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on ccirc operations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H
#define DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H

#include "mlir/Pass/Pass.h"

namespace ccirc {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "CCirc/Transforms/CCircPasses.h.inc"

} // namespace ccirc

#endif // DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H
