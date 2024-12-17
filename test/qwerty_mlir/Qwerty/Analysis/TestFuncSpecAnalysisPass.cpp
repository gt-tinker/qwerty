#include "llvm/Option/Option.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

#include "Qwerty/Analysis/FuncSpecAnalysis.h"

// Taken from TestSparseBackwardDataFlowAnalysis.cpp in the MLIR source tree:
// https://github.com/llvm/llvm-project/blob/3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff/mlir/test/lib/Analysis/DataFlow/TestSparseBackwardDataFlowAnalysis.cpp
// This is very useful for testing your analysis with FileCheck
namespace {
struct TestFuncSpecAnalysisPass
    : public mlir::PassWrapper<TestFuncSpecAnalysisPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFuncSpecAnalysisPass)

  TestFuncSpecAnalysisPass() = default;
  TestFuncSpecAnalysisPass(const TestFuncSpecAnalysisPass &other) : PassWrapper(other) {
    interprocedural = other.interprocedural;
  }

  llvm::StringRef getArgument() const override { return "test-func-spec"; }

  Option<bool> interprocedural{
      *this, "interprocedural", llvm::cl::init(true),
      llvm::cl::desc("perform interprocedural analysis")};

  void runOnOperation() override {
    mlir::Operation *root_op = getOperation();

    mlir::DataFlowSolver solver(
        mlir::DataFlowConfig().setInterprocedural(interprocedural));
    // TODO: Figure out why dead code analysis is required here. This is based
    //       on IntegerRangeAnalysis in MLIR, which says the following:
    //
    //       > This analysis depends on DeadCodeAnalysis, and will be a silent
    //       > no-op if DeadCodeAnalysis is not loaded in the same solver
    //       > context.
    //
    //       However, it is not clear to me why this is the case. Does
    //       DeadCodeAnalysis hook up dependencies properly between ops or
    //       something?
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    // TODO: Turns out SparseConstantPropagation is required as well to make
    //       analysis on scf.if work properly. I only discovered this because
    //       TestSparseBackwardDataFlowAnalysis.cpp in the MLIR tests registers
    //       it. We need to figure out why this is
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<qwerty::FuncSpecAnalysis>();
    if (mlir::failed(solver.initializeAndRun(root_op))) {
        return signalPassFailure();
    }

    llvm::raw_ostream &os = llvm::outs();
    root_op->walk([&](mlir::Operation *op) {
        auto tag = op->getAttrOfType<mlir::StringAttr>("tag");
        if (!tag) {
            return;
        }
        os << "test_tag: " << tag.getValue() << ":\n";
        for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
            os << " operand #" << index << ": ";
            const qwerty::FuncSpecLattice *specs =
                solver.lookupState<qwerty::FuncSpecLattice>(operand);
            if (specs) {
                specs->print(os);
            } else {
                os << "NULL";
            }
            os << "\n";
        }
        for (auto [index, result] : llvm::enumerate(op->getResults())) {
            os << " result #" << index << ": ";
            const qwerty::FuncSpecLattice *specs =
                solver.lookupState<qwerty::FuncSpecLattice>(result);
            if (specs) {
                specs->print(os);
            } else {
                os << "NULL";
            }
            os << "\n";
        }
    });
  }
};
} // namespace

namespace qwerty {
namespace test {
void registerTestFuncSpecAnalysisPass() {
    mlir::PassRegistration<TestFuncSpecAnalysisPass>();
}
} // namespace test
} // namespace qwerty
