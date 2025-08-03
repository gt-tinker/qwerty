//===- qwerty-opt.cpp - The qwerty-opt driver
//-------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'qwerty-opt' tool, which is the qwerty analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
// Based on CIRCT's file:
// https://github.com/llvm/circt/blob/main/tools/circt-opt/circt-opt.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Qwerty/IR/QwertyDialect.h"
#include "QCirc/IR/QCircDialect.h"
#include "CCirc/IR/CCircDialect.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "mlir/Dialect/Affine/Passes.h"

// Just for testing
namespace qwerty {
namespace test {
void registerTestFuncSpecAnalysisPass();
void registerTestQubitIndexAnalysisPass();
} // namespace test
} // namespace qwerty

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<qwerty::QwertyDialect>();
    registry.insert<qcirc::QCircDialect>();
    registry.insert<ccirc::CCircDialect>();

    // Register Dialects
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);

    // Register the standard passes we want.
    mlir::registerAllPasses();
    mlir::registerTransformsPasses(); // this should register a bunch of them

    qwerty::registerQwertyPasses();
    qcirc::registerQCircPasses();

    // Just for testing
    qwerty::test::registerTestFuncSpecAnalysisPass();
    qwerty::test::registerTestQubitIndexAnalysisPass();

    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "Qwerty modular optimizer driver", registry));
}
