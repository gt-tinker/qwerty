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

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h" // for mlir::asMainReturnCode()

#include "QCirc/IR/QCircDialect.h"

namespace {
// Yanked from mlir/lib/Target/LLVMIR/ConvertToLLVMIR.cpp, except we register
// QCirc dialect translations too below.
void registerToQIRTranslation() {
    mlir::TranslateFromMLIRRegistration registration(
        "mlir-to-qir", "Translate MLIR to QIR",
        [](mlir::Operation *op, llvm::raw_ostream &output) {
            llvm::LLVMContext llvmContext;
            auto llvmModule = mlir::translateModuleToLLVMIR(op, llvmContext);
            if (!llvmModule)
                return mlir::failure();

            llvmModule->removeDebugIntrinsicDeclarations();
            llvmModule->print(output, nullptr);
            return mlir::success();
        },
        [](mlir::DialectRegistry &registry) {
            registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
            mlir::registerAllToLLVMIRTranslations(registry);
            qcirc::registerQCircDialectTranslation(registry);
        });
}
} // namespace

int main(int argc, char **argv) {
    registerToQIRTranslation();

    return mlir::asMainReturnCode(mlir::mlirTranslateMain(
        argc, argv, "Qwerty translation driver"));
}
