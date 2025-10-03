#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Transforms/Passes.h"
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

MlirAttribute mlirIntegerAttrBigIntGet(MlirContext ctx,
                                       uint64_t bitWidth,
                                       intptr_t numValChunks,
                                       uint64_t const *valChunks) {
    llvm::ArrayRef<uint64_t> chunks(valChunks, numValChunks);
    llvm::APInt val(bitWidth, chunks);
    mlir::Type int_ty = mlir::IntegerType::get(unwrap(ctx), bitWidth);
    return wrap(mlir::IntegerAttr::get(int_ty, val));
}

MlirPass mlirCreateTransformsInlinerWithOptions(const char *options) {
    std::unique_ptr<mlir::Pass> pass = mlir::createInlinerPass();
    pass->initializeOptions(options, [](const llvm::Twine &msg) {
        llvm::errs() << "ERROR initializing inliner: " << msg << "\n";
        return mlir::failure();
    });
    return wrap(pass.release());
}
