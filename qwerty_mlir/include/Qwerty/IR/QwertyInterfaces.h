//===- QwertyInterfaces.h - Qwerty dialect interfaces/traits --*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_INTERFACES_H
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_INTERFACES_H

#include "mlir/IR/OpDefinition.h" // for TraitBase
#include "mlir/IR/PatternMatch.h" // for RewriterBase

#include "QCirc/IR/QCircInterfaces.h" // for AdjointableOpInterface
#include "Qwerty/IR/QwertyAttributes.h"

#include "Qwerty/IR/QwertyOpsTypeInterfaces.h.inc"
#include "Qwerty/IR/QwertyOpsOpInterfaces.h.inc"

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_INTERFACES_H
