#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "QCirc/Transforms/QCircPasses.h"

// Must include the declarations as they carry important visibility attributes.
#include "QCirc/Transforms/QCircPasses.capi.h.inc"

using namespace qcirc;

#ifdef __cplusplus
extern "C" {
#endif

#include "QCirc/Transforms/QCircPasses.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
