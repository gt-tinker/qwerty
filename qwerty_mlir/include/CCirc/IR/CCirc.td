//===- CCirc.td - CCirc dialect definition ---------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the CCirc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_TD

// Dialect
include "CCircDialect.td"

// Operations
include "CCircOps.td"

// Types
include "CCircTypes.td"

// Attributes
include "CCircAttributes.td"

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_TD
