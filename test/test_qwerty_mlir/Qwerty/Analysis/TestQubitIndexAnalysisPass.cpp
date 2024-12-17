#include "mlir/Pass/Pass.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Analysis/QubitIndexAnalysis.h"

// Taken from TestSparseBackwardDataFlowAnalysis.cpp in the MLIR source tree:
// https://github.com/llvm/llvm-project/blob/3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff/mlir/test/lib/Analysis/DataFlow/TestSparseBackwardDataFlowAnalysis.cpp
// This is very useful for testing your analysis with FileCheck
namespace {
struct TestQubitIndexAnalysisPass
    : public mlir::PassWrapper<TestQubitIndexAnalysisPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestQubitIndexAnalysisPass)

  llvm::StringRef getArgument() const override { return "test-qubit-index"; }

  void runOnOperation() override {
    mlir::Operation *root_op = getOperation();
    root_op->walk([&](qwerty::FuncOp func) {
        if (func.getQwertyFuncType().getReversible()) {
            mlir::Block &block = func.getBody().front();
            qwerty::QubitIndexAnalysis analysis = qwerty::runQubitIndexAnalysis(block);

            llvm::raw_ostream &os = llvm::outs();
            func->walk([&](mlir::Operation *op) {
                auto tag = op->getAttrOfType<mlir::StringAttr>("tag");
                if (!tag) {
                    return;
                }
                os << "test_tag: " << tag.getValue() << ":\n";
                for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
                    os << " operand #" << index << ": ";
                    if (analysis.count(operand)) {
                        analysis.at(operand).print(os);
                    } else {
                        os << "missing";
                    }
                    os << "\n";
                }
                for (auto [index, result] : llvm::enumerate(op->getResults())) {
                    os << " result #" << index << ": ";
                    if (analysis.count(result)) {
                        analysis.at(result).print(os);
                    } else {
                        os << "missing";
                    }
                    os << "\n";
                }
            });
        }
    });
  }
};
} // namespace

namespace qwerty {
namespace test {
void registerTestQubitIndexAnalysisPass() {
    mlir::PassRegistration<TestQubitIndexAnalysisPass>();
}
} // namespace test
} // namespace qwerty
