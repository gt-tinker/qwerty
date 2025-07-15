#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "Qwerty/Transforms/QwertyPasses.h"

// Must include the declarations as they carry important visibility attributes.
#include "Qwerty/Transforms/QwertyPasses.capi.h.inc"

using namespace qwerty;

#ifdef __cplusplus
extern "C" {
#endif

#include "Qwerty/Transforms/QwertyPasses.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
