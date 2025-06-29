//===- QCirc.td - QCirc dialect definition ---------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the QCirc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_TD
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_TD

// Dialect
include "QCircDialect.td"

// Operations
include "QCircOps.td"

// Types
include "QCircTypes.td"

// Attributes
include "QCircAttributes.td"

// Op Interfaces
include "QCircInterfaces.td"

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_TD
