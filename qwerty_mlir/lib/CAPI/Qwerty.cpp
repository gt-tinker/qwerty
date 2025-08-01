#include "mlir/CAPI/Registration.h"

#include "CAPI/Qwerty.h"
#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyTypes.h"
#include "Qwerty/IR/QwertyAttributes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Qwerty, qwerty, qwerty::QwertyDialect)

// Types

MlirType mlirQwertyFunctionTypeGet(MlirContext ctx, MlirType function_type, bool reversible) {
    return wrap(qwerty::FunctionType::get(unwrap(ctx), llvm::cast<mlir::FunctionType>(unwrap(function_type)), reversible));
}

bool mlirTypeIsAQwertyFunction(MlirType type) {
    return llvm::isa<qwerty::FunctionType>(unwrap(type));
}

MlirType mlirQwertyFunctionTypeGetFunctionType(MlirType type) {
    return wrap(llvm::cast<qwerty::FunctionType>(unwrap(type)).getFunctionType());
}

MlirType mlirQwertyBitBundleTypeGet(MlirContext ctx, uint64_t dim) {
    return wrap(qwerty::BitBundleType::get(unwrap(ctx), dim));
}

bool mlirTypeIsAQwertyBitBundle(MlirType type) {
    return llvm::isa<qwerty::BitBundleType>(unwrap(type));
}

MlirType mlirQwertyQBundleTypeGet(MlirContext ctx, uint64_t dim) {
    return wrap(qwerty::QBundleType::get(unwrap(ctx), dim));
}

bool mlirTypeIsAQwertyQBundle(MlirType type) {
    return llvm::isa<qwerty::QBundleType>(unwrap(type));
}

// Attributes

MlirAttribute mlirQwertySuperposAttrGet(
        MlirContext ctx, intptr_t numElements, MlirAttribute const *elements) {
    llvm::SmallVector<mlir::Attribute> attrs;
    (void)unwrapList(static_cast<size_t>(numElements), elements, attrs);

    llvm::SmallVector<qwerty::SuperposElemAttr> elems;
    for (mlir::Attribute attr : attrs) {
        elems.push_back(llvm::cast<qwerty::SuperposElemAttr>(attr));
    }

    return wrap(qwerty::SuperposAttr::get(unwrap(ctx), elems));
}

bool mlirAttributeIsAQwertySuperpos(MlirAttribute attr) {
    return llvm::isa<qwerty::SuperposAttr>(unwrap(attr));
}

MlirAttribute mlirQwertyBuiltinBasisAttrGet(
        MlirContext ctx, int64_t prim_basis, uint64_t dim) {
    return wrap(qwerty::BuiltinBasisAttr::get(unwrap(ctx), static_cast<qwerty::PrimitiveBasis>(prim_basis), dim));
}

bool mlirAttributeIsAQwertyBuiltinBasis(MlirAttribute attr) {
    return llvm::isa<qwerty::BuiltinBasisAttr>(unwrap(attr));
}

MlirAttribute mlirQwertySuperposElemAttrGet(
        MlirContext ctx, MlirAttribute prob, MlirAttribute phase,
        intptr_t numVectors, MlirAttribute const *vectors) {
    llvm::SmallVector<mlir::Attribute> attrs;
    (void)unwrapList(static_cast<size_t>(numVectors), vectors, attrs);

    llvm::SmallVector<qwerty::BasisVectorAttr> vecs;
    for (mlir::Attribute attr : attrs) {
        vecs.push_back(llvm::cast<qwerty::BasisVectorAttr>(attr));
    }

    return wrap(qwerty::SuperposElemAttr::get(
        unwrap(ctx),
        llvm::cast<mlir::FloatAttr>(unwrap(prob)),
        llvm::cast<mlir::FloatAttr>(unwrap(phase)),
        vecs));
}

bool mlirAttributeIsAQwertySuperposElem(MlirAttribute attr) {
    return llvm::isa<qwerty::SuperposElemAttr>(unwrap(attr));
}

MlirAttribute mlirQwertyBasisVectorAttrGet(
        MlirContext ctx, int64_t prim_basis, uint64_t dim, bool hasPhase,
        intptr_t numEigenbitChunks, uint64_t const *eigenbitChunks) {
    llvm::ArrayRef<uint64_t> chunks(eigenbitChunks, numEigenbitChunks);
    llvm::APInt eigenbits(dim, chunks);

    return wrap(qwerty::BasisVectorAttr::get(
        unwrap(ctx), static_cast<qwerty::PrimitiveBasis>(prim_basis), eigenbits, dim, hasPhase));
}

bool mlirAttributeIsAQwertyBasisVector(MlirAttribute attr) {
    return llvm::isa<qwerty::BasisVectorAttr>(unwrap(attr));
}

bool mlirQwertyBasisVectorAttrGetHasPhase(MlirAttribute attr) {
    return llvm::cast<qwerty::BasisVectorAttr>(unwrap(attr)).getHasPhase();
}

MlirAttribute mlirQwertyBasisVectorListAttrGet(
        MlirContext ctx, intptr_t numVectors, MlirAttribute const *vectors) {
    llvm::SmallVector<mlir::Attribute> attrs;
    (void)unwrapList(static_cast<size_t>(numVectors), vectors, attrs);

    llvm::SmallVector<qwerty::BasisVectorAttr> vecs;
    for (mlir::Attribute attr : attrs) {
        vecs.push_back(llvm::cast<qwerty::BasisVectorAttr>(attr));
    }

    return wrap(qwerty::BasisVectorListAttr::get(unwrap(ctx), vecs));
}

bool mlirAttributeIsAQwertyBasisVectorList(MlirAttribute attr) {
    return llvm::isa<qwerty::BasisVectorListAttr>(unwrap(attr));
}

MlirAttribute mlirQwertyBasisElemAttrGetFromVeclist(
        MlirContext ctx, MlirAttribute veclist) {
    return wrap(qwerty::BasisElemAttr::get(
        unwrap(ctx), llvm::cast<qwerty::BasisVectorListAttr>(unwrap(veclist))));
}

MlirAttribute mlirQwertyBasisElemAttrGetFromStd(
        MlirContext ctx, MlirAttribute std) {
    return wrap(qwerty::BasisElemAttr::get(
        unwrap(ctx), llvm::cast<qwerty::BuiltinBasisAttr>(unwrap(std))));
}

bool mlirAttributeIsAQwertyBasisElem(MlirAttribute attr) {
    return llvm::isa<qwerty::BasisElemAttr>(unwrap(attr));
}

MlirAttribute mlirQwertyBasisAttrGet(
        MlirContext ctx, intptr_t numElems, MlirAttribute const *elems) {
    llvm::SmallVector<mlir::Attribute> attrs;
    (void)unwrapList(static_cast<size_t>(numElems), elems, attrs);

    llvm::SmallVector<qwerty::BasisElemAttr> elemAttrs;
    for (mlir::Attribute attr : attrs) {
        elemAttrs.push_back(llvm::cast<qwerty::BasisElemAttr>(attr));
    }

    return wrap(qwerty::BasisAttr::get(unwrap(ctx), elemAttrs));
}

bool mlirAttributeIsAQwertyBasis(MlirAttribute attr) {
    return llvm::isa<qwerty::BasisAttr>(unwrap(attr));
}

uint64_t mlirQwertyBasisAttrGetDim(MlirAttribute attr) {
    return llvm::cast<qwerty::BasisAttr>(unwrap(attr)).getDim();
}
