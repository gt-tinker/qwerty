#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "CCirc/Transforms/CCircPasses.h"

// Must include the declarations as they carry important visibility attributes.
#include "CCirc/Transforms/CCircPasses.capi.h.inc"

using namespace ccirc;

#ifdef __cplusplus
extern "C" {
#endif

#include "CCirc/Transforms/CCircPasses.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
