//===- Qwerty.td - Qwerty dialect definition ---------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Qwerty dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_TD
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_TD

// Dialect
include "Qwerty/IR/QwertyDialect.td"

// Operations
include "Qwerty/IR/QwertyOps.td"

// Types
include "QwertyTypes.td"

// Attributes
include "QwertyAttributes.td"

// Op Interfaces
include "QwertyInterfaces.td"

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_TD
