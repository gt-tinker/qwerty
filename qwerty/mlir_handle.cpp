#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "QCirc/IR/QCircDialect.h"
#include "Qwerty/IR/QwertyDialect.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "QCirc/Utils/QCircUtils.h"

#include "defs.hpp"
#include "mlir_handle.hpp"
#include "qir_qrt.hpp"

namespace {
void transferModuleFlags(mlir::ModuleOp mlir_mod, llvm::Module &llvm_mod) {
    llvm::LLVMContext &ctx = llvm_mod.getContext();

    for (mlir::NamedAttribute na : mlir_mod->getDiscardableAttrs()) {
        llvm::StringRef name = na.getName().strref();
        if (name.consume_front("qcirc.flag.")) {
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
                throw JITException("Unknown module flag attribute type");
            }

            llvm_mod.addModuleFlag(llvm::Module::ModFlagBehavior::Error,
                                   name, constant);
        }
    }
}
} // namespace

MlirHandle::MlirHandle(std::string filename)
                      : context(),
                        llvmContext(),
                        builder(&context),
                        module(setupModule(builder)),
#ifdef QWERTY_USE_QIREE
                        xacc_ss(),
                        xacc(xacc_ss, /*accel_name=*/"qpp", /*shots=*/1),
                        xacc_rt(xacc_ss, xacc, /*print_accelbuf=*/false),
#endif
                        jitValid(false),
                        doFuncOpt(true),
                        irName(filename) {
    // Setup MLIR
    context.getOrLoadDialect<qcirc::QCircDialect>();
    context.getOrLoadDialect<qwerty::QwertyDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();

    mlir::DialectRegistry more_extensions;
    mlir::func::registerInlinerExtension(more_extensions);
    context.appendDialectRegistry(more_extensions);

    //// Setup LLVM
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Delete old files
    if (qwerty_debug) {
        remove_old_stage_dumps();
    }
}

std::string MlirHandle::dump_module_ir() {
    std::string ret;
    llvm::raw_string_ostream ostream(ret);
    module->print(ostream);
    return ret;
}

std::unique_ptr<llvm::Module> MlirHandle::get_qir_module(bool to_base_profile) {
    // This is a debugging tool, so go ahead and re-JIT
    invalidate_jit();
    // Hopefully something downstream will decompose our multi-controlled gates
    jit_to_gates(/*decompose_multi_ctrl=*/false);
    lower_to_llvm(to_base_profile);

    std::unique_ptr<llvm::Module> llvmModule;
    if (!(llvmModule = mlir::translateModuleToLLVMIR(*optModule, llvmContext, "qwerty"))) {
        throw JITException("Failed to translate module to LLVM IR. See errors on stderr.");
    }
    transferModuleFlags(*optModule, *llvmModule);

    // The state of the optModule is pretty cursed right now. Invalidate the
    // jit now, just in case the user calls something else
    invalidate_jit();

    if (qwerty_debug) {
        dump_stage(Stage::LlvmQir, *llvmModule);
    }

    return llvmModule;
}

std::string MlirHandle::dump_qir(bool to_base_profile) {
    std::unique_ptr<llvm::Module> llvmModule = get_qir_module(to_base_profile);
    std::string ret;
    llvm::raw_string_ostream ostream(ret);
    llvmModule->print(ostream,
                      /*AssemblyAnnotationWriter=*/nullptr,
                      /*ShouldPreserveUseListOrder=*/false,
                      /*IsForDebug=*/true);
    return ret;
}

void MlirHandle::run_optimizations() {
    mlir::PassManager pm(&context);
    // Running the canonicalizer may introduce lambdas, so run it once first
    // before the lambda lifter
    if (doFuncOpt) {
        pm.addPass(mlir::createCanonicalizerPass());
    }
    pm.addPass(qwerty::createLiftLambdasPass());
    if (doFuncOpt) {
        // Will turn qwerty.call_indirects into qwerty.calls
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createInlinerPass());
        // It seems the inliner may not run a final round of canonicalization
        // sometimes, so do it ourselves
        pm.addPass(mlir::createCanonicalizerPass());
    }
    // Remove any leftover symbols
    pm.addPass(mlir::createSymbolDCEPass());

    if (mlir::failed(pm.run(*optModule))) {
        throw JITException("Canonicalization and inlining failed. See "
                           "errors on stderr.");
    }
}

void MlirHandle::lower_to_qcirc(bool decompose_multi_ctrl) {
    mlir::PassManager pm(&context);
    // -only-pred-ones will introduce some lambdas, so lift and inline them too
    pm.addPass(qwerty::createOnlyPredOnesPass());
    pm.addPass(qwerty::createLiftLambdasPass());
    // Will turn qwerty.call_indirects into qwerty.calls
    if (doFuncOpt) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createInlinerPass());
    }
    pm.addPass(qwerty::createQwertyToQCircConversionPass());
    // Add canonicalizer pass to prune unused "builtin.unrealized_conversion_cast" ops
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(*optModule))) {
        throw JITException("Lowering to QCirc dialect failed. See errors on stderr.");
    }

    if (qwerty_debug) {
        dump_stage(Stage::InitQCirc, *optModule);
    }

    mlir::PassManager pm2(&context);
    mlir::OpPassManager &func_pm2 = pm2.nest<mlir::func::FuncOp>();
    func_pm2.addPass(qcirc::createPeepholeOptimizationPass());
    if (decompose_multi_ctrl) {
        func_pm2.addPass(qcirc::createDecomposeMultiControlPass());
        func_pm2.addPass(qcirc::createPeepholeOptimizationPass());
    }

    if (mlir::failed(pm2.run(*optModule))) {
        throw JITException("Peephole Optimizations on QCirc Dialect failed. See errors on stderr.");
    }
}

void MlirHandle::lower_to_llvm(bool to_base_profile) {
    mlir::PassManager pm(&context);
    pm.addPass(qcirc::createReplaceAnnoyingGatesPass());
    if (to_base_profile) {
        pm.addPass(qcirc::createBaseProfileModulePrepPass());
        mlir::OpPassManager &fpm = pm.nest<mlir::func::FuncOp>();
        fpm.addPass(qcirc::createBaseProfileFuncPrepPass());
    }
    pm.addPass(qcirc::createQCircToQIRConversionPass());
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(*optModule))) {
        throw JITException("Lowering to 'llvm' MLIR dialect failed. See errors on stderr.");
    }

    if (qwerty_debug) {
        dump_stage(Stage::MlirQir, *optModule);
    }

    //if (!(llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext))) {
    //    throw LoweringException("Failed to translate module to LLVM IR. See errors on stderr.");
    //}

    //unsigned optLevel = 3;  // -O level
    //unsigned sizeLevel = 0; // -Os level
    //auto optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, nullptr);
    //llvm::Error err = optPipeline(llvmModule.get());
    //if (err) {
    //    std::string errmsg;
    //    llvm::raw_string_ostream ostream(errmsg);
    //    ostream << err;
    //    throw LoweringException("LLVM optimizer failed. Error: " + errmsg);
    //}
}

void MlirHandle::jit_to_gates(bool decompose_multi_ctrl) {
    // TODO: use pass instrumentation instead of creating and running a pass
    //       manager for each phase:
    //       https://mlir.llvm.org/docs/PassManagement/#pass-instrumentation
    if (qwerty_debug) {
        dump_stage(Stage::InitQwerty, *module);
    }

    optModule = mlir::OwningOpRef<mlir::ModuleOp>(module->clone());

    run_optimizations();
    if (qwerty_debug) {
        dump_stage(Stage::OptQwerty, *optModule);
    }

    lower_to_qcirc(decompose_multi_ctrl);
    if (qwerty_debug) {
        dump_stage(Stage::OptQCirc, *optModule);
    }
}

void MlirHandle::jit() {
    assert(jit_is_invalidated()
           && "JIT is not invalidated — why are we JITing again?");
    jitValid = true;

    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

    // No need to decompose multi-controlled gates. qir-runner will take care
    // of that for us
    jit_to_gates(/*decompose_multi_ctrl=*/false);

    lower_to_llvm(/*to_base_profile=*/false);

    auto opt_pipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
    mlir::ExecutionEngineOptions engine_options;
    engine_options.transformer = opt_pipeline;

    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybe_engine =
            mlir::ExecutionEngine::create(*optModule, engine_options);

    if (!maybe_engine) {
        throw JITException("Failed to construct ExecutionEngine. "
                           "Hopefully there are details on stderr.");
    }

    exec = std::move(maybe_engine.get());
    exec->registerSymbols([=](llvm::orc::MangleAndInterner mangler) {
        #define QIR_SYMBOL(sym) {mangler(#sym), llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr((uint64_t)&sym), llvm::JITSymbolFlags::FlagNames::Callable)},
        // Generated with:
        //     grep -Eo '\<__quantum__\w+\>' qwerty/qir_qrt.hpp | awk '{ print "            QIR_SYMBOL(" $1 ")" }' | pbcopy
        return llvm::orc::SymbolMap({
            QIR_SYMBOL(__quantum__rt__initialize)
            QIR_SYMBOL(__quantum__qis__x__body)
            QIR_SYMBOL(__quantum__qis__y__body)
            QIR_SYMBOL(__quantum__qis__z__body)
            QIR_SYMBOL(__quantum__qis__h__body)
            QIR_SYMBOL(__quantum__qis__rx__body)
            QIR_SYMBOL(__quantum__qis__ry__body)
            QIR_SYMBOL(__quantum__qis__rz__body)
            QIR_SYMBOL(__quantum__qis__s__body)
            QIR_SYMBOL(__quantum__qis__s__adj)
            QIR_SYMBOL(__quantum__qis__t__body)
            QIR_SYMBOL(__quantum__qis__t__adj)
            QIR_SYMBOL(__quantum__qis__cx__body)
            QIR_SYMBOL(__quantum__qis__cy__body)
            QIR_SYMBOL(__quantum__qis__cz__body)
            QIR_SYMBOL(__quantum__qis__ccx__body)
            QIR_SYMBOL(__quantum__qis__x__ctl)
            QIR_SYMBOL(__quantum__qis__y__ctl)
            QIR_SYMBOL(__quantum__qis__z__ctl)
            QIR_SYMBOL(__quantum__qis__h__ctl)
            QIR_SYMBOL(__quantum__qis__rx__ctl)
            QIR_SYMBOL(__quantum__qis__ry__ctl)
            QIR_SYMBOL(__quantum__qis__rz__ctl)
            QIR_SYMBOL(__quantum__qis__s__ctl)
            QIR_SYMBOL(__quantum__qis__s__ctladj)
            QIR_SYMBOL(__quantum__qis__t__ctl)
            QIR_SYMBOL(__quantum__qis__t__ctladj)
            QIR_SYMBOL(__quantum__qis__m__body)
            QIR_SYMBOL(__quantum__qis__reset__body)
            QIR_SYMBOL(__quantum__rt__result_get_one)
            QIR_SYMBOL(__quantum__rt__result_equal)
            QIR_SYMBOL(__quantum__rt__qubit_allocate)
            QIR_SYMBOL(__quantum__rt__qubit_release)
            QIR_SYMBOL(__quantum__rt__array_create_1d)
            QIR_SYMBOL(__quantum__rt__array_copy)
            QIR_SYMBOL(__quantum__rt__array_update_reference_count)
            QIR_SYMBOL(__quantum__rt__array_update_alias_count)
            QIR_SYMBOL(__quantum__rt__array_get_element_ptr_1d)
            QIR_SYMBOL(__quantum__rt__array_get_size_1d)
            QIR_SYMBOL(__quantum__rt__tuple_create)
            QIR_SYMBOL(__quantum__rt__tuple_update_reference_count)
            QIR_SYMBOL(__quantum__rt__tuple_update_alias_count)
            QIR_SYMBOL(__quantum__rt__callable_create)
            QIR_SYMBOL(__quantum__rt__callable_copy)
            QIR_SYMBOL(__quantum__rt__callable_invoke)
            QIR_SYMBOL(__quantum__rt__callable_make_adjoint)
            QIR_SYMBOL(__quantum__rt__callable_make_controlled)
            QIR_SYMBOL(__quantum__rt__callable_update_reference_count)
            QIR_SYMBOL(__quantum__rt__callable_update_alias_count)
            QIR_SYMBOL(__quantum__rt__capture_update_reference_count)
            QIR_SYMBOL(__quantum__rt__capture_update_alias_count)
        });
        #undef QIR_SYMBOL
    });
}

std::string MlirHandle::qasm(std::string func_op_name, bool print_locs) {
    // We're going to trash the JIT by doing this. This is more of a
    // debugging/analysis tool than a hot/critical path, so we're not caching
    // this result. YOLO
    invalidate_jit();
    // Tools downstream may not choose optimal decompositions of
    // multi-controlled gates, so decompose them ourselves
    jit_to_gates(/*decompose_multi_ctrl=*/true);

    mlir::func::FuncOp func = optModule->lookupSymbol<mlir::func::FuncOp>(func_op_name);

    if (!func) {
        throw JITException("No such function " + func_op_name + " in "
                           "optimized MLIR");
    }

    // Spam stderr with errors
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

    std::string result;
    if (mlir::failed(qcirc::generateQasm(func, print_locs, result))) {
        throw JITException("Could not generate QASM. With any luck, there are "
                           "details on stderr.");
    }
    return result;
}

void MlirHandle::dump_stage(Stage stage, std::function<void(llvm::raw_ostream &)> print_func) {
    std::string filename = irName + stage_file_name(stage);
    std::ofstream stream(filename, std::ios::app);
    stream << "// " << stage_name(stage) << "\n";
    {
        llvm::raw_os_ostream llvm_stream(stream);
        print_func(llvm_stream);
    }
    stream.close();
}

void MlirHandle::dump_stage(Stage stage, mlir::ModuleOp module) {
    dump_stage(stage, [&](llvm::raw_ostream &ostream) {
        module->print(ostream);
    });
}

void MlirHandle::dump_stage(Stage stage, llvm::Module &module) {
    dump_stage(stage, [&](llvm::raw_ostream &ostream) {
        module.print(ostream,
                     /*AssemblyAnnotationWriter=*/nullptr,
                     /*ShouldPreserveUseListOrder=*/false,
                     /*IsForDebug=*/true);
    });
}

void MlirHandle::remove_old_stage_dumps() {
#ifdef QWERTY_USE_QIREE
    std::remove(qiree_log_filename);
#endif

    for (int i = 0; i < static_cast<int>(Stage::_Count); i++) {
        Stage stage = static_cast<Stage>(i);
        std::string path = irName + stage_file_name(stage);
        std::remove(path.c_str());
    }
}
