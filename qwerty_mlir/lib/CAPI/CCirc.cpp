#include "mlir/CAPI/Registration.h"

#include "CAPI/CCirc.h"
#include "CCirc/IR/CCircDialect.h"
#include "CCirc/IR/CCircTypes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CCirc, ccirc, ccirc::CCircDialect)

// Types

MlirType mlirCCircWireTypeGet(MlirContext ctx, uint64_t dim) {
    return wrap(ccirc::WireType::get(unwrap(ctx), dim));
}

bool mlirTypeIsACCircWire(MlirType type) {
    return llvm::isa<ccirc::WireType>(unwrap(type));
}
