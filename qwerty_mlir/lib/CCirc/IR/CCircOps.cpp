//===- CCircOps.cpp - CCirc dialect ops --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include <unordered_set>
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "CCirc/IR/CCircOps.h"
#include "CCirc/IR/CCircDialect.h"
#include "CCirc/Transforms/CCircPasses.h"

#define GET_OP_CLASSES
#include "CCirc/IR/CCircOps.cpp.inc"