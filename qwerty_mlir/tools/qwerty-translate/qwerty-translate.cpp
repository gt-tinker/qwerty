//===- qwerty-translate.cpp - The qwerty-translate driver
//-------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'qwerty-translate' tool, which is the qwerty analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
// Based on CIRCT's file:
// https://github.com/llvm/circt/blob/main/tools/circt-translate/circt-translate.cpp
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h" // for mlir::asMainReturnCode()

#include "QCirc/IR/QCircDialect.h"

extern llvm::cl::opt<bool> WriteNewDbgInfoFormat;

namespace {
void registerToQIRTranslation() {
    mlir::TranslateFromMLIRRegistration registration(
        "mlir-to-qir", "Translate MLIR to QIR",
        [](mlir::Operation *op, llvm::raw_ostream &output) {
            llvm::LLVMContext llvmContext;
            auto llvmModule = mlir::translateModuleToLLVMIR(op, llvmContext);
            if (!llvmModule) {
                return mlir::failure();
            }

            // Taken from mlir/lib/Target/LLVMIR/ConvertToLLVMIR.cpp
            // See https://llvm.org/docs/RemoveDIsDebugInfo.html
            llvm::ScopedDbgInfoFormatSetter formatSetter(*llvmModule,
                                                WriteNewDbgInfoFormat);

            if (WriteNewDbgInfoFormat)
                llvmModule->removeDebugIntrinsicDeclarations();
            llvmModule->print(output, nullptr);
            return mlir::success();
        },
        [](mlir::DialectRegistry &registry) {
            mlir::registerBuiltinDialectTranslation(registry);
            mlir::registerLLVMDialectTranslation(registry);
            qcirc::registerQCircDialectTranslation(registry);
        });
}
} // namespace

int main(int argc, char **argv) {
    registerToQIRTranslation();

    return mlir::asMainReturnCode(mlir::mlirTranslateMain(
        argc, argv, "Qwerty translation driver"));
}
