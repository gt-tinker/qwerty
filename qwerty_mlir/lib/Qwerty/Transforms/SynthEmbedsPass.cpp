#include <queue>

#include "mlir/IR/SymbolTable.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "CCirc/IR/CCircOps.h"
#include "PassDetail.h"

// This is a pass that converts all qwerty.embed_* ops that reference classical
// circuits into qwerty.func_const ops referencing qwerty.funcs that contain
// quantum circuits.

namespace {

struct SynthEmbedsPass
        : public qwerty::SynthEmbedsBase<SynthEmbedsPass> {
    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        mlir::SymbolTable symbol_table(mod);

        mlir::IRRewriter rewriter(&getContext());
        llvm::SmallVector<ccirc::CircuitOp> worklist;
        for (ccirc::CircuitOp circ :
                mod.getBodyRegion().getOps<ccirc::CircuitOp>()) {
            worklist.push_back(circ);
        }

        while (!worklist.empty()) {
            ccirc::CircuitOp circ = worklist.back();
            worklist.pop_back();

            // TODO: find all users of this circ's symbol
            // TODO: for each each embed, gather embed kinds needed
            // TODO: create embed functions
            // TODO: replace every embed with a func const
            // TODO: delete circuitop
            // TODO: pass pipeline should run symbol dce afterward to get rid of circuits

            //size_t next_id = 0;
            //// Copy all the lambdas into a temporary vector since the iterator
            //// seems to be invalidated when we modify the IR
            //llvm::SmallVector<qwerty::LambdaOp> lambdas;
            //funcop->walk([&](qwerty::LambdaOp lambda) {
            //    lambdas.push_back(lambda);
            //});

            //for (qwerty::LambdaOp lambda : lambdas) {
            //    std::string basename = funcop.getSymName().str() + "__lambda";
            //    std::string new_func_name;
            //    do {
            //        new_func_name = basename + std::to_string(next_id++);
            //    } while (symbol_table.lookup(new_func_name));

            //    rewriter.setInsertionPoint(funcop);
            //    qwerty::FuncOp new_funcop =
            //        rewriter.create<qwerty::FuncOp>(lambda.getLoc(),
            //            new_func_name, lambda.getResult().getType());
            //    new_funcop.setPrivate();

            //    rewriter.inlineRegionBefore(
            //        lambda.getRegion(), new_funcop.getBody(),
            //        new_funcop.getBody().begin());

            //    rewriter.setInsertionPoint(lambda);
            //    rewriter.replaceOpWithNewOp<qwerty::FuncConstOp>(
            //        lambda, new_funcop, lambda.getCaptures());
            //    worklist.push(new_funcop);
            //}
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qwerty::createSynthEmbedsPass() {
    return std::make_unique<SynthEmbedsPass>();
}
