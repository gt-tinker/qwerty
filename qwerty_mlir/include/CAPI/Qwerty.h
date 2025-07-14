#ifndef QWERTY_MLIR_C_DIALECT_QWERTY_H
#define QWERTY_MLIR_C_DIALECT_QWERTY_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Qwerty, qwerty);

// Types

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

// Attributes

/// Creates an qwerty::SuperposAttr containing the given list of
/// qwerty::SuperposElemAttrs.
MLIR_CAPI_EXPORTED MlirAttribute mlirQwertySuperposAttrGet(
    MlirContext ctx, intptr_t numElements, MlirAttribute const *elements);

/// Returns true if this is a qwerty::SuperposAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertySuperpos(MlirAttribute attr);

/// Creates an qwerty::SuperposElemAttr containing the given list of
/// qwerty::BasisVectorAttrs.
MLIR_CAPI_EXPORTED MlirAttribute mlirQwertySuperposElemAttrGet(
        MlirContext ctx, MlirAttribute prob, MlirAttribute phase,
        intptr_t numVectors, MlirAttribute const *vectors);

/// Returns true if this is a qwerty::SuperposElemAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertySuperposElem(MlirAttribute attr);

/// Creates an qwerty::BasisVectorAttr. The eigenbits are passed as llvm::APInt
/// expects in its "bigVal" constructor, which is little endian.
MLIR_CAPI_EXPORTED MlirAttribute mlirQwertyBasisVectorAttrGet(
        MlirContext ctx, int64_t prim_basis, uint64_t dim, bool hasPhase,
        intptr_t numEigenbitChunks, uint64_t const *eigenbitChunks);

/// Returns true if this is a qwerty::BasisVectorAttr
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertyBasisVector(MlirAttribute attr);

/// Creates an qwerty::BasisVectorListAttr containing the given list of
/// qwerty::BasisVectorAttrs.
MLIR_CAPI_EXPORTED MlirAttribute mlirQwertyBasisVectorListAttrGet(
        MlirContext ctx, intptr_t numVectors, MlirAttribute const *vectors);

/// Returns true if this is a qwerty::BasisVectorListAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertyBasisVectorList(MlirAttribute attr);

/// Creates an qwerty::BasisElemAttr from the qwerty::BasisVectorListAttr
/// provided.
MLIR_CAPI_EXPORTED MlirAttribute mlirQwertyBasisElemAttrGetFromVeclist(
        MlirContext ctx, MlirAttribute veclist);

/// Returns true if this is a qwerty::BasisElemAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertyBasisElem(MlirAttribute attr);

/// Creates an qwerty::BasisAttr containing the given list of
/// qwerty::BasisVectorAttrs.
MLIR_CAPI_EXPORTED MlirAttribute mlirQwertyBasisAttrGet(
        MlirContext ctx, intptr_t numElems, MlirAttribute const *elems);

/// Returns true if this is a qwerty::BasisAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertyBasis(MlirAttribute attr);

/// Calls qwerty::BasisAttr::getDim()
MLIR_CAPI_EXPORTED uint64_t mlirQwertyBasisAttrGetDim(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // QWERTY_MLIR_C_DIALECT_QWERTY_H
