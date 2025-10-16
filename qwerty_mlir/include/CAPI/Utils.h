#ifndef QWERTY_MLIR_C_UTILS_H
#define QWERTY_MLIR_C_UTILS_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
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

/// Allows getting an arbitrarily-sized IntegerAttr. The numValChunks and
/// valChunks provided must be of the format expected by llvm::APInt's bigVal
/// constructor.
MLIR_CAPI_EXPORTED MlirAttribute mlirIntegerAttrBigIntGet(
    MlirContext ctx,
    uint64_t bitWidth,
    intptr_t numValChunks,
    uint64_t const *valChunks);

MLIR_CAPI_EXPORTED MlirPass mlirCreateTransformsInlinerWithOptions(
    const char *options);

#ifdef __cplusplus
}
#endif

#endif // QWERTY_MLIR_C_UTILS_H
