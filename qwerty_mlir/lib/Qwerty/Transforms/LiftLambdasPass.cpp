#include <queue>

#include "mlir/IR/SymbolTable.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "PassDetail.h"

// This is a pass that converts all qwerty.lambda ops into qwerty.func_const
// ops referencing qwerty.funcs.

namespace {

struct LiftLambdasPass
        : public qwerty::LiftLambdasBase<LiftLambdasPass> {
    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();
        mlir::SymbolTable symbol_table(mod);

        mlir::IRRewriter rewriter(&getContext());
        std::queue<qwerty::FuncOp> worklist;
        for (qwerty::FuncOp func :
                mod.getBodyRegion().getOps<qwerty::FuncOp>()) {
            worklist.push(func);
        }

        while (!worklist.empty()) {
            qwerty::FuncOp funcop = worklist.front();
            worklist.pop();

            size_t next_id = 0;
            // Copy all the lambdas into a temporary vector since the iterator
            // seems to be invalidated when we modify the IR
            llvm::SmallVector<qwerty::LambdaOp> lambdas;
            funcop->walk([&](qwerty::LambdaOp lambda) {
                lambdas.push_back(lambda);
            });

            for (qwerty::LambdaOp lambda : lambdas) {
                std::string basename = funcop.getSymName().str() + "__lambda";
                std::string new_func_name;
                do {
                    new_func_name = basename + std::to_string(next_id++);
                } while (symbol_table.lookup(new_func_name));

                rewriter.setInsertionPoint(funcop);
                qwerty::FuncOp new_funcop =
                    rewriter.create<qwerty::FuncOp>(lambda.getLoc(),
                        new_func_name, lambda.getResult().getType());
                new_funcop.setPrivate();

                rewriter.inlineRegionBefore(
                    lambda.getRegion(), new_funcop.getBody(),
                    new_funcop.getBody().begin());

                rewriter.setInsertionPoint(lambda);
                rewriter.replaceOpWithNewOp<qwerty::FuncConstOp>(
                    lambda, new_funcop, lambda.getCaptures());
                worklist.push(new_funcop);
            }
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qwerty::createLiftLambdasPass() {
    return std::make_unique<LiftLambdasPass>();
}
