#include "mlir/CAPI/Registration.h"

#include "CAPI/Qwerty.h"
#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyTypes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Qwerty, qwerty, qwerty::QwertyDialect)

MlirType mlirQwertyFunctionTypeGet(MlirContext ctx, MlirType function_type, bool reversible) {
    return wrap(qwerty::FunctionType::get(unwrap(ctx), llvm::cast<mlir::FunctionType>(unwrap(function_type)), reversible));
}

MlirType mlirQwertyBitBundleTypeGet(MlirContext ctx, uint64_t dim) {
    return wrap(qwerty::BitBundleType::get(unwrap(ctx), dim));
}
