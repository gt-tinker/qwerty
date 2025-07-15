#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "CAPI/Utils.h"

void mlirRegisterInlinerExtensions(MlirDialectRegistry registry) {
    mlir::DialectRegistry *reg = unwrap(registry);
    mlir::func::registerInlinerExtension(*reg);
    mlir::LLVM::registerInlinerInterface(*reg);
}

void mlirExecutionEngineRegisterSymbols(
        MlirExecutionEngine jit,
        intptr_t numEntries,
        MlirSymbolMapEntry const *entries) {
    unwrap(jit)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;

        for (intptr_t i = 0; i < numEntries; i++) {
            symbolMap[interner(unwrap(entries[i].symbolName))] = {
                llvm::orc::ExecutorAddr::fromPtr(entries[i].addr),
                llvm::JITSymbolFlags(
                    static_cast<llvm::JITSymbolFlags::FlagNames>(entries[i].jitSymbolFlags))
            };
        }

        return symbolMap;
    });
}

