#include "CCirc/IR/CCircOps.h"
#include "CCirc/Transforms/CCircPasses.h"

#include "PassDetail.h"

namespace {

struct CCircToXAGConversionPass
        : public ccirc::CCircToXAGConversionBase<CCircToXAGConversionPass> {
    void runOnOperation() override {
        ccirc::CircuitOp circ = getOperation();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> ccirc::createCCircToXAGConversionPass() {
    return std::make_unique<CCircToXAGConversionPass>();
}
