//===- CCircOps.h - CCirc dialect ops -----------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_H
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/CommonFolders.h"

#include "CCirc/IR/CCircDialect.h"
#include "CCirc/IR/CCircTypes.h"
#include "CCirc/IR/CCircAttributes.h"

#define GET_OP_CLASSES
#include "CCirc/IR/CCircOps.h.inc"

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_H
