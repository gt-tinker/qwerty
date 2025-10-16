#include "llvm-c/Types.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "CAPI/Utils.h"

void mlirRegisterInlinerExtensions(MlirDialectRegistry registry) {
    mlir::DialectRegistry &reg = *unwrap(registry);
    mlir::func::registerInlinerExtension(reg);
    mlir::LLVM::registerInlinerInterface(reg);
}

void mlirRegisterLLVMIRTranslations(MlirDialectRegistry registry) {
    mlir::DialectRegistry &reg = *unwrap(registry);
    mlir::registerBuiltinDialectTranslation(reg);
    mlir::registerLLVMDialectTranslation(reg);
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
    mlir::LogicalResult res = pass->initializeOptions(options, [](const llvm::Twine &msg) {
        llvm::errs() << "ERROR initializing inliner: " << msg << "\n";
        return mlir::failure();
    });
    // TODO: better error handling... somehow
    assert(res.succeeded() && "Initializing pass options failed");
    return wrap(pass.release());
}

void mlirTransferModuleFlags(
        MlirOperation mlir_module, LLVMModuleRef llvm_module) {
    mlir::ModuleOp mlir_mod = llvm::cast<mlir::ModuleOp>(unwrap(mlir_module));
    llvm::Module *llvm_mod = llvm::unwrap(llvm_module);
    llvm::LLVMContext &ctx = llvm_mod->getContext();

    for (mlir::NamedAttribute na : mlir_mod->getDiscardableAttrs()) {
        llvm::StringRef name = na.getName().strref();
        if (name.consume_front("llvm.flag.")) {
            mlir::Attribute attr = na.getValue();
            // Very limited supported conversions, but will do the trick for now
            llvm::Constant *constant;
            if (mlir::BoolAttr bool_attr =
                    llvm::dyn_cast<mlir::BoolAttr>(attr)) {
                constant = llvm::ConstantInt::getBool(ctx, bool_attr.getValue());
            } else if (mlir::IntegerAttr int_attr =
                    llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
                constant = llvm::ConstantInt::get(ctx, int_attr.getValue());
            } else {
                // TODO: better error handling
                assert(0 && "Missing handling for module flag conversion");
            }

            llvm_mod->addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                    name, constant);
        }
    }
}
