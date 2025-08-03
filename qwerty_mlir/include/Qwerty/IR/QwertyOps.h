//===- QwertyOps.h - Qwerty dialect ops -----------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_OPS_H
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/CommonFolders.h"

#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyTypes.h"
#include "Qwerty/IR/QwertyAttributes.h"
#include "Qwerty/IR/QwertyInterfaces.h"
#include "QCirc/IR/QCircTypes.h"
#include "QCirc/IR/QCircInterfaces.h"
#include "CCirc/IR/CCircOps.h"

#define GET_OP_CLASSES
#include "Qwerty/IR/QwertyOps.h.inc"

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_OPS_H
