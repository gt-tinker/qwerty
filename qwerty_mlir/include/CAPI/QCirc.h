#ifndef QWERTY_MLIR_C_DIALECT_QCIRC_H
#define QWERTY_MLIR_C_DIALECT_QCIRC_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QCirc, qcirc);

#ifdef __cplusplus
}
#endif

#include "QCirc/Transforms/QCircPasses.capi.h.inc"

#endif // QWERTY_MLIR_C_DIALECT_QCIRC_H
