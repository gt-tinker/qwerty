#include "mlir/CAPI/Registration.h"

#include "CAPI/Qwerty.h"
#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyTypes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Qwerty, qwerty, qwerty::QwertyDialect)

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
