#ifndef QWERTY_MLIR_C_DIALECT_CCIRC_H
#define QWERTY_MLIR_C_DIALECT_CCIRC_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CCirc, ccirc);

// Types

/// Creates a ccirc::WireType
MLIR_CAPI_EXPORTED MlirType mlirCCircWireTypeGet(MlirContext ctx,
                                                 uint64_t dim);

/// Returns true if this is a ccirc::WireType
MLIR_CAPI_EXPORTED bool mlirTypeIsACCircWire(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // QWERTY_MLIR_C_DIALECT_CCIRC_H
