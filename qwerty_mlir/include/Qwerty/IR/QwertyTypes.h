//===- QwertyTypes.h - Qwerty dialect types -------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_TYPES_H
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include "QCirc/IR/QCircInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "Qwerty/IR/QwertyOpsTypes.h.inc"

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_TYPES_H
