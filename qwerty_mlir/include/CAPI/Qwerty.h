#ifndef QWERTY_MLIR_C_DIALECT_QWERTY_H
#define QWERTY_MLIR_C_DIALECT_QWERTY_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Qwerty, qwerty);

/// Creates a qwerty::FunctionType
MLIR_CAPI_EXPORTED MlirType mlirQwertyFunctionTypeGet(MlirContext ctx,
                                                      MlirType function_type,
                                                      bool reversible);

/// Returns true if this is a qwerty::FunctionType
MLIR_CAPI_EXPORTED bool mlirTypeIsAQwertyFunction(MlirType type);

/// Returns the inner mlir::FunctionType
MLIR_CAPI_EXPORTED MlirType mlirQwertyFunctionTypeGetFunctionType(MlirType type);

/// Creates a qwerty::BitBundleType
MLIR_CAPI_EXPORTED MlirType mlirQwertyBitBundleTypeGet(MlirContext ctx,
                                                       uint64_t dim);

/// Returns true if this is a qwerty::BitBundleType
MLIR_CAPI_EXPORTED bool mlirTypeIsAQwertyBitBundle(MlirType type);

/// Creates a qwerty::QBundleType
MLIR_CAPI_EXPORTED MlirType mlirQwertyQBundleTypeGet(MlirContext ctx,
                                                     uint64_t dim);

/// Returns true if this is a qwerty::QBundleType
MLIR_CAPI_EXPORTED bool mlirTypeIsAQwertyQBundle(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // QWERTY_MLIR_C_DIALECT_QWERTY_H
