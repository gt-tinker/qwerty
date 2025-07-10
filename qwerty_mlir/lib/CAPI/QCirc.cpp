#include "mlir/CAPI/Registration.h"

#include "CAPI/QCirc.h"
#include "QCirc/IR/QCircDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QCirc, qcirc, qcirc::QCircDialect)
