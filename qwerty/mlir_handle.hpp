#ifndef MLIR_H
#define MLIR_H

#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#ifdef QWERTY_USE_QIREE
#include "qirxacc/XaccQuantum.hh"
#include "qirxacc/XaccTupleRuntime.hh"
#endif

// This is the class in charge of managing the MLIRContext and root ModuleOp.
// It also is in charge of running all PassManagers used in the Qwerty
// compiler.

// Stage of JIT compilation
enum class Stage {
    InitQwerty,
    OptQwerty,
    InitQCirc,
    OptQCirc,
    MlirQir,
    LlvmQir,
    _Count,
};

static inline std::string stage_name(Stage stage) {
    switch (stage) {
    case Stage::InitQwerty: return "Initial Qwerty-Dialect IR"; break;
    case Stage::OptQwerty: return "Optimized Qwerty-Dialect IR"; break;
    case Stage::InitQCirc: return "Initial QCirc-Dialect IR"; break;
    case Stage::OptQCirc: return "Optimized QCirc-Dialect IR"; break;
    case Stage::MlirQir: return "Final LLVM-Dialect IR (QIR)"; break;
    case Stage::LlvmQir: return "Final LLVM IR (QIR)"; break;
    default: assert(0 && "Missing name of Stage"); return "";
    }
}

static inline std::string stage_file_name(Stage stage) {
    switch (stage) {
    case Stage::InitQwerty: return "_qwerty_init.mlir"; break;
    case Stage::OptQwerty: return "_qwerty_opt.mlir"; break;
    case Stage::InitQCirc: return "_qcirc_init.mlir"; break;
    case Stage::OptQCirc: return "_qcirc_opt.mlir"; break;
    case Stage::MlirQir: return "_qir.mlir"; break;
    case Stage::LlvmQir: return "_qir.ll"; break;
    default: assert(0 && "Missing name of Stage"); return "";
    }
}

struct MlirHandle {
#ifdef QWERTY_USE_QIREE
    static constexpr const char *qiree_log_filename = "qiree.log";
#endif

    mlir::MLIRContext context;
    llvm::LLVMContext llvmContext;
    mlir::OpBuilder builder;
    // Two modules are maintained here: `module' is our pristine copy that
    // QpuLoweringVisitor writes Qwerty IR to. Then when we run passes (called
    // "jitting" in this file), we copy it over to optModule, which is our
    // dirty copy that eventually becomes the QCirc and then llvm dialect.
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::OwningOpRef<mlir::ModuleOp> optModule;
    std::unique_ptr<mlir::ExecutionEngine> exec;
    // See docs/qiree.md for info on QIR-EE integration
#ifdef QWERTY_USE_QIREE
    std::stringstream xacc_ss;
    qiree::XaccQuantum xacc;
    qiree::XaccTupleRuntime xacc_rt;
#endif
    bool jitValid;
    bool doFuncOpt;
    std::string irName;

    static mlir::OwningOpRef<mlir::ModuleOp> setupModule(mlir::OpBuilder &builder) {
        return mlir::ModuleOp::create(builder.getUnknownLoc());
    }

    MlirHandle(std::string filename);
    std::string dump_module_ir();
    std::unique_ptr<llvm::Module> get_qir_module(bool to_base_profile);
    std::string dump_qir(bool to_base_profile);
    void run_optimizations();
    void lower_to_qcirc(bool decompose_multi_ctrl);
    void lower_to_llvm(bool to_base_profile);
    void jit_to_gates(bool decompose_multi_ctrl);
    void jit();
    std::string qasm(std::string func_op_name, bool print_locs);
    void dump_stage(Stage stage, mlir::ModuleOp module);
    void dump_stage(Stage stage, llvm::Module &module);
    void dump_stage(Stage stage, std::function<void(llvm::raw_ostream &)> print_func);
    void remove_old_stage_dumps();
    void invalidate_jit() {
        exec.reset();
        // This will erase() the ModuleOp on the left-hand side
        optModule = mlir::OwningOpRef<mlir::ModuleOp>();
        jitValid = false;
    }
    inline bool jit_is_invalidated() { return !jitValid; }
    inline void jit_if_needed() { if (jit_is_invalidated()) { jit(); } }
    inline void set_func_opt(bool do_func_opt) { doFuncOpt = do_func_opt; }
};

// IR optimization is done across the whole mlir Module, so there is no
// DebugInfo to use to print a line number. When this is thrown, it's almost
// always a compiler bug.
struct JITException : public std::runtime_error {
    std::string message;

    JITException(std::string message)
                : std::runtime_error("JITException"),
                  message(message) {}
};

#endif
