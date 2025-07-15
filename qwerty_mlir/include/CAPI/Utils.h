#ifndef QWERTY_MLIR_C_UTILS_H
#define QWERTY_MLIR_C_UTILS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/ExecutionEngine.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterInlinerExtensions(MlirDialectRegistry registry);

typedef struct {
    MlirStringRef symbolName;
    void *addr;
    uint8_t jitSymbolFlags; // llvm::JitSymbolFlags
} MlirSymbolMapEntry;

MLIR_CAPI_EXPORTED void mlirExecutionEngineRegisterSymbols(
    MlirExecutionEngine jit,
    intptr_t numEntries,
    MlirSymbolMapEntry const *entries);

#ifdef __cplusplus
}
#endif

#endif // QWERTY_MLIR_C_UTILS_H
