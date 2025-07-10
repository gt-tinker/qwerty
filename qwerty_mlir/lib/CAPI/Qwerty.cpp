#include "mlir/CAPI/Registration.h"

#include "CAPI/Qwerty.h"
#include "Qwerty/IR/QwertyDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Qwerty, qwerty, qwerty::QwertyDialect)
