//===- QCircToQIRConversionPass.cpp - Lower to LLVM IR -----------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircTypes.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "PassDetail.h"
#include "RewriterDetail.h"

// This pass is responsible for converting QCirc-dialect MLIR to llvm-dialect
// MLIR. Besides some moderately interesting pre-processing, what you will find
// here is pretty tedious construction of llvm-dialect ops. In general, there
// will be a dialect conversion pattern for every QCirc Op. Your best bet at
// navigating this file will be starting at the bottom with the pass itself and
// jumping backwards to conversion patterns for particular QCirc ops you care
// about.

namespace {

// Add some extra LLVM-specific attributes to functions
void addExtraLLVMAttrsToFuncs(mlir::ModuleOp module_op) {
    for (mlir::func::FuncOp func : module_op.getBodyRegion()
                                            .getOps<mlir::func::FuncOp>()) {
        mlir::SymbolTable::Visibility vis =
            mlir::SymbolTable::getSymbolVisibility(func);
        if (vis == mlir::SymbolTable::Visibility::Private) {
            // In the Qwerty compiler, symbol visibility is the same as
            // linkage. But FuncToLLVM doesn't know about that if we don't set
            // the necessary discardable attribute on each FuncOp.
            func->setDiscardableAttr(
                "llvm.linkage",
                mlir::LLVM::LinkageAttr::get(
                    module_op.getContext(),
                    mlir::LLVM::linkage::Linkage::Internal));
        } else if (vis == mlir::SymbolTable::Visibility::Public) {
            // This tells LLVM to produce wrapper code needed to call this
            // function with ExecutionEngine::invokePacked().
            func->setDiscardableAttr(
                "llvm.emit_c_interface",
                mlir::UnitAttr::get(module_op->getContext()));
        }
    }
}

// Avoid confusing copyAndFreeHeapAllocatedObjects() by just inlining
// qcirc.calcs now (they might contain arrpacks of classical bits)
void inlineCalcs(mlir::ModuleOp module_op) {
    llvm::SmallVector<qcirc::CalcOp> calcs_to_inline;
    module_op->walk([&](qcirc::CalcOp calc) {
        calcs_to_inline.push_back(calc);
    });

    mlir::IRRewriter rewriter(module_op.getContext());
    for (qcirc::CalcOp calc : calcs_to_inline) {
        assert(calc.getRegion().hasOneBlock());
        mlir::Block &calc_block = calc.getRegion().front();
        qcirc::CalcYieldOp calc_yield = llvm::cast<qcirc::CalcYieldOp>(calc_block.getTerminator());
        rewriter.inlineBlockBefore(&calc_block, calc, calc.getInputs());
        rewriter.replaceOp(calc, calc_yield.getResults());
        rewriter.eraseOp(calc_yield);
    }
}

bool isRefCountedType(mlir::Type ty) {
    // TODO: Replace this with a type trait maybe?
    return llvm::isa<qcirc::ArrayType>(ty)
           || llvm::isa<qcirc::CallableType>(ty);
}

mlir::Value copyRefCounted(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value rc_val) {
    mlir::Type rc_ty = rc_val.getType();
    mlir::Value copied;

    if (llvm::isa<qcirc::ArrayType>(rc_ty)) {
        copied = builder.create<qcirc::ArrayCopyOp>(loc, rc_val)
                        .getArrayOut();
    } else if (llvm::isa<qcirc::CallableType>(rc_ty)) {
        copied = builder.create<qcirc::CallableCopyOp>(loc, rc_val)
                        .getCallableOut();
    } else {
        assert(0 && "copyRefCounted() called on non-refcounted value");
        return nullptr;
    }

    return copied;
}

// This must be called pre-conversion, when rc_val represents an edge coming
// from a qcirc dialect op (not an llvm dialect one)
void freeRefCounted(mlir::OpBuilder &builder,
                    mlir::Value rc_val) {
    mlir::Location loc = rc_val.getLoc();
    mlir::Type rc_ty = rc_val.getType();

    if (llvm::isa<qcirc::ArrayType>(rc_ty)) {
        builder.create<qcirc::ArrayFreeOp>(loc, rc_val);
    } else if (llvm::isa<qcirc::CallableType>(rc_ty)) {
        builder.create<qcirc::CallableFreeOp>(loc, rc_val);
    } else {
        assert(0 && "freeRefCounted() called on non-refcounted value");
    }
}

size_t getNumCaptures(mlir::ModuleOp module_op, mlir::StringAttr str) {
    mlir::Region *region = &(module_op.getRegion());
    auto func_sym_uses_optional =
        mlir::SymbolTable::getSymbolUses(str, region);
    assert(func_sym_uses_optional.has_value()
           && "Missing uses from symbol table, bug!");
    for (const mlir::SymbolTable::SymbolUse &sym_use
            : func_sym_uses_optional.value()) {
        if (qcirc::CallableMetadataOp meta = llvm::dyn_cast<qcirc::CallableMetadataOp>(sym_use.getUser())) {
            return meta.getCaptureTypes().size();
        }
    }

    // If no metadata, then this cannot be built into a callable and can only
    // be called directly. So treat captures (even if they exist) as ordinary
    // arguments, since that is harmless
    return 0;
}

// Our IR is pass-by-value and assumes all values are stack-allocated, but that
// is not the way QIR works. Instead, in QIR, arrays and callables are
// heap-allocated, so we need to free them manually by decrementing their
// reference counts. We could ignore this reality, but then the QIR we produce
// would leak like a stuck pig.
//
// Instead, this subroutine updates the IR such that the following invariants
// are maintained:
//
// 1. Every block argument that has the type of a heap-allocated object
//    receives a fresh, clean, beautiful copy of the object with a reference
//    count of 1.
// 2. It is the job of the callee (or whoever transfers control flow to that
//    block) to copy it if needed to maintain invariant #1.
// 3. Every heap-allocated object must be freed at the end of the block where
//    it is defined. This includes block arguments.
//
// (There is a little optimization to skip #2 and #3 if it's safe to do so, but
//  don't let that confuse you.)
//
// This is not the most efficient approach, but at least correctness is
// maintained, and this code rarely runs because inlining works quite well.
void copyAndFreeHeapAllocatedObjects(mlir::ModuleOp module_op) {
    mlir::OpBuilder builder(module_op.getContext());

    for (mlir::func::FuncOp func : module_op.getBodyRegion()
                                            .getOps<mlir::func::FuncOp>()) {
        // Build a set of all candidates to free, and start up a queue of
        // operands to copy
        llvm::SmallVector<std::reference_wrapper<mlir::OpOperand>> to_copy;
        llvm::SmallPtrSet<mlir::Value, 4> to_free;

        func->walk([&](mlir::Operation *op) {
            if (auto pack = llvm::dyn_cast<qcirc::ArrayPackOp>(op)) {
                // TODO: could something other than qubits or bits be packed
                //       here?
                to_free.insert(pack.getArray());
            } else if (auto create =
                    llvm::dyn_cast<qcirc::CallableCreateOp>(op)) {
                for (mlir::OpOperand &operand : create.getCapturesMutable()) {
                    to_copy.emplace_back(operand);
                }
                to_free.insert(create.getCallable());
            } else if (auto adj =
                    llvm::dyn_cast<qcirc::CallableAdjointOp>(op)) {
                to_copy.emplace_back(adj.getCallableInMutable());
                to_free.insert(adj.getCallableOut());
            } else if (auto ctrl =
                    llvm::dyn_cast<qcirc::CallableControlOp>(op)) {
                to_copy.emplace_back(ctrl.getCallableInMutable());
                to_free.insert(ctrl.getCallableOut());
            } else if (auto call = llvm::dyn_cast<mlir::func::CallOp>(op)) {
                mlir::StringAttr str = mlir::StringAttr::get(module_op.getContext(), call.getCallee());
                size_t num_args_skip = getNumCaptures(module_op,str);
                size_t operand_index = 0;
                for (mlir::OpOperand &operand : call.getOperandsMutable()) {
                    if(operand_index >= num_args_skip) {
                        to_copy.emplace_back(operand);
                    }
                    operand_index += 1;
                }
                to_free.insert(call->result_begin(), call->result_end());
            } else if (auto calli =
                    llvm::dyn_cast<qcirc::CallableInvokeOp>(op)) {
                for (mlir::OpOperand &operand
                        : calli.getCallOperandsMutable()) {
                    to_copy.emplace_back(operand);
                }
                to_free.insert(calli.getResults().begin(),
                               calli.getResults().end());
            // The result passed to both returns and yields should be copied
            } else if (auto ret =
                    llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
                for (mlir::OpOperand &ret_operand : ret.getOperandsMutable()) {
                    to_copy.emplace_back(ret_operand);
                }
            } else if (auto yield =
                    llvm::dyn_cast<mlir::scf::YieldOp>(op)) {
                for (mlir::OpOperand &yield_result :
                        yield.getResultsMutable()) {
                    to_copy.emplace_back(yield_result);
                }
            // ...And thus the result of an scf.if may need to be freed
            } else if (auto if_ =
                    llvm::dyn_cast<mlir::scf::IfOp>(op)) {
                to_free.insert(if_.getResults().begin(),
                               if_.getResults().end());
            }
        });

        assert(func.getBody().hasOneBlock());
        mlir::Block &func_block = func.getBody().front();
        // Imagine arguments need to be freed too
        size_t num_args_skip = getNumCaptures(module_op, func.getSymNameAttr());
        to_free.insert(func_block.args_begin()+num_args_skip, func_block.args_end());

        // Next step: actually mutate the IR
        for (mlir::OpOperand &operand : to_copy) {
            mlir::Value cur_val = operand.get();
            if (!isRefCountedType(cur_val.getType())) {
                continue;
            }
            if (cur_val.hasOneUse() && to_free.contains(cur_val)) {
                // Humble optimization: if this is the only use and we were
                // about to free anyone, elide the copy and the free
                to_free.erase(cur_val);
            } else {
                // Otherwise, we need to copy. Oh well.
                mlir::Operation *owner = operand.getOwner();
                builder.setInsertionPoint(owner);
                mlir::Value new_val = copyRefCounted(
                    builder, owner->getLoc(), cur_val);
                operand.set(new_val);
            }
        }

        for (mlir::Value alloced : to_free) {
            if (!isRefCountedType(alloced.getType())) {
                continue;
            }
            // This is subtle but crucial: values should be free()'d at the end
            // of the same block where they are defined. For example, if they
            // are defined in a function, we should free() before the return,
            // but if they are defined in a branch of an if-statement, we
            // should free at the end of that branch. (We cannot free when
            // the overall function returns in the latter case because the
            // branch in question may not dominate the end of the function.)
            mlir::Block *containing_block = alloced.getParentBlock();
            builder.setInsertionPoint(containing_block->getTerminator());
            freeRefCounted(builder, alloced);
        }
    }
}

// Whether __quantum__qis__cname() is a (usually) thing in QIR
inline bool qirSupportsC(qcirc::Gate1Q gate) {
    return gate == qcirc::Gate1Q::X
           || gate == qcirc::Gate1Q::Y
           || gate == qcirc::Gate1Q::Z;
}
// Whether __quantum__qis__ccname() is a (usually) thing in QIR
inline bool qirSupportsCC(qcirc::Gate1Q gate) {
    return gate == qcirc::Gate1Q::X;
}

mlir::LLVM::GlobalOp createStringConstant(
        mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef sym_name,
        llvm::StringRef text) {
    size_t n_bytes = text.size() + 1; // null terminator
    mlir::Type type = builder.getType<mlir::LLVM::LLVMArrayType>(
        builder.getI8Type(), n_bytes);

    llvm::SmallVector<int8_t> bytes(text.begin(), text.end());
    bytes.push_back(0x00);
    mlir::Attribute array = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get({static_cast<int64_t>(n_bytes)},
                                    builder.getI8Type()),
        bytes);

    mlir::LLVM::GlobalOp global = builder.create<mlir::LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true,
        mlir::LLVM::Linkage::Internal, sym_name, array);
    global.setPrivate();
    return global;
}

class QCircToQIRTypeConverter : public mlir::LLVMTypeConverter {
public:
    QCircToQIRTypeConverter(mlir::MLIRContext *context) : mlir::LLVMTypeConverter(context) {
        // Now that LLVM uses opaque pointers, this type converter is sure a
        // whole lot less interesting
        addConversion([=](qcirc::QubitType type) {
            return mlir::LLVM::LLVMPointerType::get(context);
        });
        addConversion([=](qcirc::ArrayType type) {
            return mlir::LLVM::LLVMPointerType::get(context);
        });
        addConversion([=](qcirc::CallableType type) {
            return mlir::LLVM::LLVMPointerType::get(context);
        });
    }
};

struct GateStubs {
    mlir::LLVM::LLVMFuncOp stub_1q;
    mlir::LLVM::LLVMFuncOp stub_c1q;
    mlir::LLVM::LLVMFuncOp stub_cc1q;
    mlir::LLVM::LLVMFuncOp stub_mc1q;
};

// Giant struct holding pointers to pre-created func ops for prototypes for QIR
// intrinsics. Why not create these on the fly in the conversion patterns
// below? Well, technically it is a big no-no to modify IR (including the
// containing ModuleOp) outside the peephole you promised to convert. So
// instead we scan the IR before conversion to find out what QIR intrinsic
// prototypes we'll need and then create them upfront (see
// IntrinsicStubsFactory below). The reason to do this scan rather than
// speculatively creating all prototypes is that it dramatically bloats the
// resulting QIR to the point of near-unreadability. Overall, this approach is
// not going to bring a reader of this code to tears of joy, but it's safe and
// relatively simple.
struct IntrinsicStubs {
    // Stubs
    llvm::DenseMap<qcirc::Gate1Q, GateStubs> gates1q;
    llvm::DenseMap<qcirc::Gate1Q1P, GateStubs> gates1q1p;
    llvm::DenseMap<qcirc::Gate2Q, mlir::LLVM::LLVMFuncOp> gates2q;
    mlir::LLVM::LLVMFuncOp qalloc;
    mlir::LLVM::LLVMFuncOp qfree;
    mlir::LLVM::LLVMFuncOp reset;
    mlir::LLVM::LLVMFuncOp measure;
    mlir::LLVM::LLVMFuncOp resultGetOne;
    mlir::LLVM::LLVMFuncOp resultEqual;
    mlir::LLVM::LLVMFuncOp create1dArray;
    mlir::LLVM::LLVMFuncOp gep1dArray;
    mlir::LLVM::LLVMFuncOp size1dArray;
    mlir::LLVM::LLVMFuncOp arrayUpdateRc;
    mlir::LLVM::LLVMFuncOp arrayUpdateAlias;
    mlir::LLVM::LLVMFuncOp arrayCopy;
    mlir::LLVM::LLVMFuncOp tupleCreate;
    mlir::LLVM::LLVMFuncOp tupleUpdateRc;
    mlir::LLVM::LLVMFuncOp tupleUpdateAlias;
    mlir::LLVM::LLVMFuncOp initialize;
    mlir::LLVM::LLVMFuncOp measureInPlace;
    mlir::LLVM::LLVMFuncOp tupleRecord;
    mlir::LLVM::LLVMFuncOp resultRecord;
    mlir::LLVM::LLVMFuncOp callableCreate;
    mlir::LLVM::LLVMFuncOp callableInvoke;
    mlir::LLVM::LLVMFuncOp callableCopy;
    mlir::LLVM::LLVMFuncOp callableAdjoint;
    mlir::LLVM::LLVMFuncOp callableControl;
    mlir::LLVM::LLVMFuncOp callableUpdateRc;
    mlir::LLVM::LLVMFuncOp callableUpdateAlias;
    mlir::LLVM::LLVMFuncOp callableCaptureUpdateRc;
    mlir::LLVM::LLVMFuncOp callableCaptureUpdateAlias;
    mlir::LLVM::LLVMFuncOp fail;
    // Messages
    mlir::LLVM::GlobalOp badSpecMsg;

    IntrinsicStubs(llvm::DenseMap<qcirc::Gate1Q, GateStubs> gates1q,
                   llvm::DenseMap<qcirc::Gate1Q1P, GateStubs> gates1q1p,
                   llvm::DenseMap<qcirc::Gate2Q, mlir::LLVM::LLVMFuncOp> gates2q,
                   mlir::LLVM::LLVMFuncOp qalloc,
                   mlir::LLVM::LLVMFuncOp qfree,
                   mlir::LLVM::LLVMFuncOp reset,
                   mlir::LLVM::LLVMFuncOp measure,
                   mlir::LLVM::LLVMFuncOp resultGetOne,
                   mlir::LLVM::LLVMFuncOp resultEqual,
                   mlir::LLVM::LLVMFuncOp create1dArray,
                   mlir::LLVM::LLVMFuncOp gep1dArray,
                   mlir::LLVM::LLVMFuncOp size1dArray,
                   mlir::LLVM::LLVMFuncOp arrayUpdateRc,
                   mlir::LLVM::LLVMFuncOp arrayUpdateAlias,
                   mlir::LLVM::LLVMFuncOp arrayCopy,
                   mlir::LLVM::LLVMFuncOp tupleCreate,
                   mlir::LLVM::LLVMFuncOp tupleUpdateRc,
                   mlir::LLVM::LLVMFuncOp tupleUpdateAlias,
                   mlir::LLVM::LLVMFuncOp initialize,
                   mlir::LLVM::LLVMFuncOp measureInPlace,
                   mlir::LLVM::LLVMFuncOp tupleRecord,
                   mlir::LLVM::LLVMFuncOp resultRecord,
                   mlir::LLVM::LLVMFuncOp callableCreate,
                   mlir::LLVM::LLVMFuncOp callableInvoke,
                   mlir::LLVM::LLVMFuncOp callableCopy,
                   mlir::LLVM::LLVMFuncOp callableAdjoint,
                   mlir::LLVM::LLVMFuncOp callableControl,
                   mlir::LLVM::LLVMFuncOp callableUpdateRc,
                   mlir::LLVM::LLVMFuncOp callableUpdateAlias,
                   mlir::LLVM::LLVMFuncOp callableCaptureUpdateRc,
                   mlir::LLVM::LLVMFuncOp callableCaptureUpdateAlias,
                   mlir::LLVM::LLVMFuncOp fail,
                   mlir::LLVM::GlobalOp badSpecMsg) :
        gates1q(gates1q),
        gates1q1p(gates1q1p),
        gates2q(gates2q),
        qalloc(qalloc),
        qfree(qfree),
        reset(reset),
        measure(measure),
        resultGetOne(resultGetOne),
        resultEqual(resultEqual),
        create1dArray(create1dArray),
        gep1dArray(gep1dArray),
        size1dArray(size1dArray),
        arrayUpdateRc(arrayUpdateRc),
        arrayUpdateAlias(arrayUpdateAlias),
        arrayCopy(arrayCopy),
        tupleCreate(tupleCreate),
        tupleUpdateRc(tupleUpdateRc),
        tupleUpdateAlias(tupleUpdateAlias),
        initialize(initialize),
        measureInPlace(measureInPlace),
        tupleRecord(tupleRecord),
        resultRecord(resultRecord),
        callableCreate(callableCreate),
        callableInvoke(callableInvoke),
        callableCopy(callableCopy),
        callableAdjoint(callableAdjoint),
        callableControl(callableControl),
        callableUpdateRc(callableUpdateRc),
        callableUpdateAlias(callableUpdateAlias),
        callableCaptureUpdateRc(callableCaptureUpdateRc),
        callableCaptureUpdateAlias(callableCaptureUpdateAlias),
        fail(fail),
        badSpecMsg(badSpecMsg) {}
};

// See the comment above for IntrinsicStubs for what this is. The complicated
// handling for x versus cx versus ccx below is ugly here in the compiler, but
// it produces dramatically cleaner QIR.
struct IntrinsicStubsFactory {
    mlir::ModuleOp module;
    mlir::OpBuilder builder;

    IntrinsicStubsFactory(mlir::ModuleOp module,
                          mlir::MLIRContext *context) :
        module(module),
        builder(context) {}

    mlir::LLVM::LLVMFuncOp stubIntrinsic(llvm::StringRef name, mlir::LLVM::LLVMFunctionType funcType) {
        builder.setInsertionPointToEnd(module.getBody());
        return builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, funcType);
    }

    mlir::LLVM::GlobalOp createMessage(llvm::StringRef sym_name, llvm::StringRef message) {
        builder.setInsertionPointToEnd(module.getBody());
        return createStringConstant(builder, builder.getUnknownLoc(), sym_name, message);
    }

    llvm::DenseMap<qcirc::Gate1Q, GateStubs> stub1QGates(bool &found_multi_ctrl_out) {
        bool found_multi_ctrl = false;
        llvm::DenseMap<qcirc::Gate1Q, GateStubs> gates1q;

        auto gate1qFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(),
                                                                builder.getType<mlir::LLVM::LLVMPointerType>());
        auto gatec1qFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(),
                                                                 {builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                  builder.getType<mlir::LLVM::LLVMPointerType>()});
        auto gatecc1qFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(),
                                                                  {builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                   builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                   builder.getType<mlir::LLVM::LLVMPointerType>()});
        // This is actually the same as above because of LLVM no longer having typed pointers
        auto gatemc1qFuncType = gatec1qFuncType;

        module->walk([&](qcirc::Gate1QOp gate_op) {
            if (gate_op.getControls().empty()
                    && (!gates1q.count(gate_op.getGate()) || !gates1q[gate_op.getGate()].stub_1q)) {
                std::string lowerName = stringifyGate1Q(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__";
                if (lowerName.size() > 2
                        && lowerName[lowerName.size()-2] == 'd'
                        && lowerName[lowerName.size()-1] == 'g') {
                    intrinsicName += lowerName.substr(0, lowerName.size()-2) + "__adj";
                } else {
                    intrinsicName += lowerName + "__body";
                }
                gates1q[gate_op.getGate()].stub_1q = stubIntrinsic(intrinsicName, gate1qFuncType);
            } else if (gate_op.getControls().size() == 1
                    && qirSupportsC(gate_op.getGate())
                    && (!gates1q.count(gate_op.getGate()) || !gates1q[gate_op.getGate()].stub_c1q)) {
                std::string lowerName = stringifyGate1Q(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__c" + lowerName + "__body";
                gates1q[gate_op.getGate()].stub_c1q = stubIntrinsic(intrinsicName, gatec1qFuncType);
            } else if (gate_op.getControls().size() == 2
                    && qirSupportsCC(gate_op.getGate())
                    && (!gates1q.count(gate_op.getGate()) || !gates1q[gate_op.getGate()].stub_cc1q)) {
                std::string lowerName = stringifyGate1Q(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__cc" + lowerName + "__body";
                gates1q[gate_op.getGate()].stub_cc1q = stubIntrinsic(intrinsicName, gatecc1qFuncType);
            } else if ((gate_op.getControls().size() > 2
                    || (gate_op.getControls().size() == 1 && !qirSupportsC(gate_op.getGate()))
                    || (gate_op.getControls().size() == 2 && !qirSupportsCC(gate_op.getGate())))
                    && (!gates1q.count(gate_op.getGate()) || !gates1q[gate_op.getGate()].stub_mc1q)) {
                found_multi_ctrl = true;
                std::string lowerName = stringifyGate1Q(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__";
                if (lowerName.size() > 2
                        && lowerName[lowerName.size()-2] == 'd'
                        && lowerName[lowerName.size()-1] == 'g') {
                    intrinsicName += lowerName.substr(0, lowerName.size()-2) + "__ctladj";
                } else {
                    intrinsicName += lowerName + "__ctl";
                }
                gates1q[gate_op.getGate()].stub_mc1q = stubIntrinsic(intrinsicName, gatemc1qFuncType);
            }
        });

        found_multi_ctrl_out = found_multi_ctrl;
        return gates1q;
    }

    llvm::DenseMap<qcirc::Gate1Q1P, GateStubs> stub1Q1PGates(bool &found_multi_ctrl_out) {
        bool found_multi_ctrl = false;
        llvm::DenseMap<qcirc::Gate1Q1P, GateStubs> gates1q1p;

        auto gate1q1pFuncType = mlir::LLVM::LLVMFunctionType::get(
                builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getF64Type(),
                                                              builder.getType<mlir::LLVM::LLVMPointerType>()});
        auto gatemc1q1pFuncType = mlir::LLVM::LLVMFunctionType::get(
                builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                              builder.getType<mlir::LLVM::LLVMPointerType>()});

        module->walk([&](qcirc::Gate1Q1POp gate_op) {
            if (gate_op.getControls().empty()
                    && (!gates1q1p.count(gate_op.getGate()) || !gates1q1p[gate_op.getGate()].stub_1q)) {
                std::string lowerName = stringifyGate1Q1P(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__" + lowerName + "__body";
                gates1q1p[gate_op.getGate()].stub_1q = stubIntrinsic(intrinsicName, gate1q1pFuncType);
            } else if (!gate_op.getControls().empty()
                    && (!gates1q1p.count(gate_op.getGate()) || !gates1q1p[gate_op.getGate()].stub_mc1q)) {
                found_multi_ctrl = true;
                std::string lowerName = stringifyGate1Q1P(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__" + lowerName + "__ctl";
                gates1q1p[gate_op.getGate()].stub_mc1q = stubIntrinsic(intrinsicName, gatemc1q1pFuncType);
            }
        });

        found_multi_ctrl_out = found_multi_ctrl;
        return gates1q1p;
    }

    llvm::DenseMap<qcirc::Gate2Q, mlir::LLVM::LLVMFuncOp> stub2QGates() {
        llvm::DenseMap<qcirc::Gate2Q, mlir::LLVM::LLVMFuncOp> gates2q;

        auto gate2qFuncType = mlir::LLVM::LLVMFunctionType::get(
                builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getType<mlir::LLVM::LLVMPointerType>()});
        module->walk([&](qcirc::Gate2QOp gate_op) {
            if (gate_op.getControls().empty()
                    && !gates2q.count(gate_op.getGate())) {
                std::string lowerName = stringifyGate2Q(gate_op.getGate()).lower();
                std::string intrinsicName = "__quantum__qis__" + lowerName + "__body";
                gates2q[gate_op.getGate()] = stubIntrinsic(intrinsicName, gate2qFuncType);
            }
            // SWAPs with controls should've been handled by ReplaceAnnoyingGatesPass
        });

        return gates2q;
    }

    template <typename T>
    bool exists() {
        mlir::WalkResult res = module->walk([](T op) {
            return mlir::WalkResult::interrupt();
        });
        return res.wasInterrupted();
    }

    template <typename T>
    bool existsWhere(std::function<bool(T)> pred) {
        mlir::WalkResult res = module->walk([&](T op) {
            if (pred(op)) {
                return mlir::WalkResult::interrupt();
            } else {
                return mlir::WalkResult::advance();
            }
        });
        return res.wasInterrupted();
    }

    IntrinsicStubs build() {
        // TODO: use a bitmask for all these op types instead of all the
        //       linear-time exists<>() searching

        mlir::LLVM::LLVMFuncOp qallocStub;
        if (exists<qcirc::QallocOp>()) {
            auto qallocFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMPointerType>(), {});
            qallocStub = stubIntrinsic("__quantum__rt__qubit_allocate", qallocFuncType);
        }

        mlir::LLVM::LLVMFuncOp qfreeStub, resetStub;
        if (exists<qcirc::QfreeOp>() || exists<qcirc::QfreeZeroOp>()) {
            auto qfreeFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(),
                                                                   builder.getType<mlir::LLVM::LLVMPointerType>());
            qfreeStub = stubIntrinsic("__quantum__rt__qubit_release", qfreeFuncType);
        }

        if (exists<qcirc::QfreeOp>()) {
            auto resetFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMVoidType>(),
                                                                   builder.getType<mlir::LLVM::LLVMPointerType>());
            resetStub = stubIntrinsic("__quantum__qis__reset__body", resetFuncType);
        }

        mlir::LLVM::LLVMFuncOp measureStub, getOneStub, equalStub;
        if (exists<qcirc::MeasureOp>()) {
            auto measureFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                     builder.getType<mlir::LLVM::LLVMPointerType>());
            measureStub = stubIntrinsic("__quantum__qis__m__body", measureFuncType);

            auto getOneFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMPointerType>(), {});
            getOneStub = stubIntrinsic("__quantum__rt__result_get_one", getOneFuncType);

            auto equalFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getI1Type(), {builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                                         builder.getType<mlir::LLVM::LLVMPointerType>()});
            equalStub = stubIntrinsic("__quantum__rt__result_equal", equalFuncType);
        }

        bool found_multi_ctrl_1q = false;
        bool found_multi_ctrl_1q1p = false;

        auto gates1q = stub1QGates(found_multi_ctrl_1q);
        auto gates1q1p = stub1Q1PGates(found_multi_ctrl_1q1p);
        auto gates2q = stub2QGates();

        mlir::LLVM::LLVMFuncOp create1dArrayStub, gep1dArrayStub,
                               arrayUpdateRcStub, arrayCopyStub;
        if (found_multi_ctrl_1q || found_multi_ctrl_1q1p
                || exists<qcirc::ArrayPackOp>()) {
            auto create1dArrayFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                           {builder.getI32Type(),
                                                                            builder.getI64Type()});
            create1dArrayStub = stubIntrinsic("__quantum__rt__array_create_1d", create1dArrayFuncType);
        }
        if (found_multi_ctrl_1q || found_multi_ctrl_1q1p
                || exists<qcirc::ArrayPackOp>()
                || exists<qcirc::ArrayUnpackOp>()) {
            auto gep1dArrayFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMPointerType>(),
                    {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI64Type()});
            gep1dArrayStub = stubIntrinsic("__quantum__rt__array_get_element_ptr_1d", gep1dArrayFuncType);
        }
        if (exists<qcirc::ArrayCopyOp>()) {
            auto arrayCopyFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMPointerType>(),
                    {builder.getType<mlir::LLVM::LLVMPointerType>(),
                     builder.getIntegerType(1)});
            arrayCopyStub = stubIntrinsic("__quantum__rt__array_copy", arrayCopyFuncType);
        }
        auto metaHasCaptures = [](qcirc::CallableMetadataOp meta) {
            return !meta.getCaptureTypes().empty();
        };
        if (found_multi_ctrl_1q || found_multi_ctrl_1q1p
                || exists<qcirc::ArrayFreeOp>()
                || existsWhere<qcirc::CallableMetadataOp>(metaHasCaptures)) {
            auto arrayUpdateRcFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            arrayUpdateRcStub = stubIntrinsic("__quantum__rt__array_update_reference_count", arrayUpdateRcFuncType);
        }

        mlir::LLVM::LLVMFuncOp tupleCreateStub, tupleUpdateRcStub;
        if (found_multi_ctrl_1q1p
                || exists<qcirc::CallableCreateOp>()
                || exists<qcirc::CallableInvokeOp>()) {
            auto tupleCreateFuncType = mlir::LLVM::LLVMFunctionType::get(builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI64Type());
            tupleCreateStub = stubIntrinsic("__quantum__rt__tuple_create", tupleCreateFuncType);
        }
        if (found_multi_ctrl_1q1p
                || exists<qcirc::CallableMetadataOp>()
                || exists<qcirc::CallableInvokeOp>()) {
            auto tupleUpdateRcFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            tupleUpdateRcStub = stubIntrinsic("__quantum__rt__tuple_update_reference_count", tupleUpdateRcFuncType);
        }

        // The following four are needed for the base profile

        mlir::LLVM::LLVMFuncOp initializeStub;
        if (exists<qcirc::InitOp>()) {
            auto initializeFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), builder.getType<mlir::LLVM::LLVMPointerType>());
            initializeStub = stubIntrinsic("__quantum__rt__initialize", initializeFuncType);
        }

        mlir::LLVM::LLVMFuncOp measureInPlaceStub, tupleRecordStub,
                               resultRecordStub;
        if (exists<qcirc::UglyMeasureOp>()) {
            auto measureInPlaceFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                  builder.getType<mlir::LLVM::LLVMPointerType>()});
            measureInPlaceStub = stubIntrinsic("__quantum__qis__mz__body", measureInPlaceFuncType);
        }
        if (exists<qcirc::UglyRecordOp>()) {
            auto tupleRecordFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getI64Type(),
                                                                  builder.getType<mlir::LLVM::LLVMPointerType>()});
            tupleRecordStub = stubIntrinsic("__quantum__rt__tuple_record_output", tupleRecordFuncType);

            auto resultRecordFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(),
                                                                  builder.getType<mlir::LLVM::LLVMPointerType>()});
            resultRecordStub = stubIntrinsic("__quantum__rt__result_record_output", resultRecordFuncType);
        }

        // Callables

        mlir::LLVM::LLVMFuncOp arrayUpdateAliasStub, size1dArrayStub,
                               tupleUpdateAliasStub, callableCreateStub,
                               callableInvokeStub, callableCopyStub,
                               callableAdjointStub, callableControlStub,
                               callableUpdateRcStub, callableUpdateAliasStub,
                               callableCaptureUpdateRcStub,
                               callableCaptureUpdateAliasStub, failStub;
        mlir::LLVM::GlobalOp badSpecMsg;

        if (existsWhere<qcirc::CallableMetadataOp>(metaHasCaptures)) {
            auto arrayUpdateAliasFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            arrayUpdateAliasStub = stubIntrinsic("__quantum__rt__array_update_alias_count", arrayUpdateAliasFuncType);

            auto tupleUpdateAliasFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            tupleUpdateAliasStub = stubIntrinsic("__quantum__rt__tuple_update_alias_count", tupleUpdateAliasFuncType);

            auto callableUpdateAliasFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            callableUpdateAliasStub = stubIntrinsic("__quantum__rt__callable_update_alias_count", callableUpdateAliasFuncType);

            auto callableCaptureUpdateAliasFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            callableCaptureUpdateAliasStub = stubIntrinsic("__quantum__rt__capture_update_alias_count", callableCaptureUpdateAliasFuncType);
        }
        if (exists<qcirc::CallableMetadataOp>()
                || exists<qcirc::CallableFreeOp>()) {
            auto callableUpdateRcFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            callableUpdateRcStub = stubIntrinsic("__quantum__rt__callable_update_reference_count", callableUpdateRcFuncType);

            auto callableCaptureUpdateRcFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(), {builder.getType<mlir::LLVM::LLVMPointerType>(), builder.getI32Type()});
            callableCaptureUpdateRcStub = stubIntrinsic("__quantum__rt__capture_update_reference_count", callableCaptureUpdateRcFuncType);
        }
        if (exists<qcirc::CallableCopyOp>()) {
            auto callableCopyFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMPointerType>(),
                    {builder.getType<mlir::LLVM::LLVMPointerType>(),
                     builder.getIntegerType(1)});
            callableCopyStub = stubIntrinsic("__quantum__rt__callable_copy", callableCopyFuncType);
        }
        auto hasControlledSpec = [](qcirc::CallableMetadataOp meta) {
            return llvm::any_of(meta.getSpecsRange(), [](qcirc::FuncSpecAttr spec) {
                return spec.getNumControls();
            });
        };
        if (existsWhere<qcirc::CallableMetadataOp>(hasControlledSpec)) {
            auto size1dArrayFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getI64Type(), builder.getType<mlir::LLVM::LLVMPointerType>());
            size1dArrayStub = stubIntrinsic("__quantum__rt__array_get_size_1d", size1dArrayFuncType);

            auto failFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(),
                    builder.getType<mlir::LLVM::LLVMPointerType>());
            failStub = stubIntrinsic("__quantum__rt__fail", failFuncType);
            badSpecMsg = createMessage("__bad_spec_msg", "Bad specialization called");
        }
        if (exists<qcirc::CallableCreateOp>()) {
            auto callableCreateFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMPointerType>(),
                    {builder.getType<mlir::LLVM::LLVMPointerType>(),
                     builder.getType<mlir::LLVM::LLVMPointerType>(),
                     builder.getType<mlir::LLVM::LLVMPointerType>()});
            callableCreateStub = stubIntrinsic("__quantum__rt__callable_create", callableCreateFuncType);
        }
        if (exists<qcirc::CallableAdjointOp>()) {
            auto callableAdjointFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(),
                    builder.getType<mlir::LLVM::LLVMPointerType>());
            callableAdjointStub = stubIntrinsic("__quantum__rt__callable_make_adjoint", callableAdjointFuncType);
        }
        if (exists<qcirc::CallableControlOp>()) {
            auto callableControlFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(),
                    builder.getType<mlir::LLVM::LLVMPointerType>());
            callableControlStub = stubIntrinsic("__quantum__rt__callable_make_controlled", callableControlFuncType);
        }
        if (exists<qcirc::CallableInvokeOp>()) {
            auto callableInvokeFuncType = mlir::LLVM::LLVMFunctionType::get(
                    builder.getType<mlir::LLVM::LLVMVoidType>(),
                    {builder.getType<mlir::LLVM::LLVMPointerType>(),
                     builder.getType<mlir::LLVM::LLVMPointerType>(),
                     builder.getType<mlir::LLVM::LLVMPointerType>()});
            callableInvokeStub = stubIntrinsic("__quantum__rt__callable_invoke", callableInvokeFuncType);
        }

        IntrinsicStubs stubs(gates1q, gates1q1p, gates2q,
                             qallocStub, qfreeStub,
                             resetStub, measureStub,
                             getOneStub, equalStub,
                             create1dArrayStub, gep1dArrayStub,
                             size1dArrayStub, arrayUpdateRcStub,
                             arrayUpdateAliasStub, arrayCopyStub,
                             tupleCreateStub, tupleUpdateRcStub,
                             tupleUpdateAliasStub,
                             initializeStub, measureInPlaceStub,
                             tupleRecordStub, resultRecordStub,
                             callableCreateStub, callableInvokeStub,
                             callableCopyStub, callableAdjointStub,
                             callableControlStub, callableUpdateRcStub,
                             callableUpdateAliasStub,
                             callableCaptureUpdateRcStub,
                             callableCaptureUpdateAliasStub, failStub,
                             badSpecMsg);
        return stubs;
    }
};

mlir::Value sizeofHack(mlir::Location loc, mlir::OpBuilder &builder, mlir::Type T, mlir::Type sizeType) {
    // Clever hack to get calculate sizeof(T) in LLVM IR
    // Based on: https://stackoverflow.com/a/30830445/321301
    mlir::Value hack = builder.create<mlir::LLVM::GEPOp>(
            loc,
            builder.getType<mlir::LLVM::LLVMPointerType>(),
            T,
            builder.create<mlir::LLVM::ZeroOp>(loc, builder.getType<mlir::LLVM::LLVMPointerType>()).getRes(),
            std::initializer_list<mlir::LLVM::GEPArg>{1}).getRes();
    mlir::Value sizeof_T = builder.create<mlir::LLVM::PtrToIntOp>(loc, sizeType, hack).getRes();
    return sizeof_T;
}

struct Gate1QOpLowering : public mlir::OpConversionPattern<qcirc::Gate1QOp> {
    IntrinsicStubs &stubs;

    Gate1QOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::Gate1QOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1QOp gate_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        GateStubs &gate_stubs = stubs.gates1q[gate_op.getGate()];

        if (gate_op.getControls().empty()) {
            rewriter.create<mlir::LLVM::CallOp>(gate_op.getLoc(), gate_stubs.stub_1q, adaptor.getQubit());
            // Replace any uses with the qubit we took as input
            rewriter.replaceOp(gate_op, adaptor.getQubit());
            return mlir::success();
        } else if (gate_op.getControls().size() == 1 && qirSupportsC(gate_op.getGate())) {
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    gate_stubs.stub_c1q,
                    mlir::ValueRange({adaptor.getControls()[0], adaptor.getQubit()}));
            // Replace any uses with the qubits we took as input
            rewriter.replaceOp(gate_op, {adaptor.getControls()[0], adaptor.getQubit()});
            return mlir::success();
        } else if (gate_op.getControls().size() == 2 && qirSupportsCC(gate_op.getGate())) {
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    gate_stubs.stub_cc1q,
                    mlir::ValueRange({adaptor.getControls()[0],
                                      adaptor.getControls()[1],
                                      adaptor.getQubit()}));
            // Replace any uses with the qubits we took as input
            rewriter.replaceOp(gate_op, {adaptor.getControls()[0],
                                         adaptor.getControls()[1],
                                         adaptor.getQubit()});
            return mlir::success();
        } else { // gate_op.getControls().size() > 1
            // sizeof(void *)
            mlir::Value ptr_size = sizeofHack(
                    gate_op.getLoc(),
                    rewriter,
                    rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                    rewriter.getI32Type());

            // Create array for control qubits
            mlir::Value n_controls = rewriter.create<mlir::arith::ConstantOp>(
                    gate_op.getLoc(),
                    rewriter.getI64IntegerAttr(gate_op.getControls().size())).getResult();
            mlir::Value controls_arr = rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    stubs.create1dArray,
                    std::initializer_list<mlir::Value>{ptr_size, n_controls}).getResult();

            // Add all the controls into the array
            for (size_t i = 0; i < gate_op.getControls().size(); i++) {
                mlir::Value i_val = rewriter.create<mlir::arith::ConstantOp>(
                        gate_op.getLoc(),
                        rewriter.getI64IntegerAttr(i)).getResult();
                mlir::Value elem_ptr = rewriter.create<mlir::LLVM::CallOp>(
                        gate_op.getLoc(),
                        stubs.gep1dArray,
                        std::initializer_list<mlir::Value>{controls_arr, i_val}).getResult();
                rewriter.create<mlir::LLVM::StoreOp>(gate_op.getLoc(), adaptor.getControls()[i], elem_ptr);
            }

            // Finally, invoke the multi-control stub!
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    gate_stubs.stub_mc1q,
                    mlir::ValueRange({controls_arr, adaptor.getQubit()}));

            // free() the array
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    stubs.arrayUpdateRc,
                    std::initializer_list<mlir::Value>{
                        controls_arr,
                        rewriter.create<mlir::arith::ConstantOp>(gate_op.getLoc(), rewriter.getI32IntegerAttr(-1)).getResult()});

            // Replace any uses with the qubits we took as input
            llvm::SmallVector<mlir::Value> new_outputs;
            new_outputs.append(adaptor.getControls().begin(), adaptor.getControls().end());
            new_outputs.push_back(adaptor.getQubit());
            rewriter.replaceOp(gate_op, new_outputs);
            return mlir::success();
        }
    }
};

struct Gate1Q1POpLowering : public mlir::OpConversionPattern<qcirc::Gate1Q1POp> {
    IntrinsicStubs &stubs;

    Gate1Q1POpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::Gate1Q1POp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::Gate1Q1POp gate_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        GateStubs &gate_stubs = stubs.gates1q1p[gate_op.getGate()];

        if (gate_op.getControls().empty()) {
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    gate_stubs.stub_1q,
                    std::initializer_list<mlir::Value>{adaptor.getParam(), adaptor.getQubit()});
            // Replace any uses with the qubit we took as input
            rewriter.replaceOp(gate_op, adaptor.getQubit());
            return mlir::success();
        } else { // gate_op.getControls().size() > 1
            // sizeof(void *)
            mlir::Value ptr_size = sizeofHack(
                    gate_op.getLoc(),
                    rewriter,
                    rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                    rewriter.getI32Type());

            // Create array for control qubits
            mlir::Value n_controls = rewriter.create<mlir::arith::ConstantOp>(
                    gate_op.getLoc(),
                    rewriter.getI64IntegerAttr(gate_op.getControls().size())).getResult();
            mlir::Value controls_arr = rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    stubs.create1dArray,
                    std::initializer_list<mlir::Value>{ptr_size, n_controls}).getResult();

            // Add all the controls into the array
            for (size_t i = 0; i < gate_op.getControls().size(); i++) {
                mlir::Value i_val = rewriter.create<mlir::arith::ConstantOp>(
                        gate_op.getLoc(),
                        rewriter.getI64IntegerAttr(i)).getResult();
                mlir::Value elem_ptr = rewriter.create<mlir::LLVM::CallOp>(
                        gate_op.getLoc(),
                        stubs.gep1dArray,
                        std::initializer_list<mlir::Value>{controls_arr, i_val}).getResult();
                rewriter.create<mlir::LLVM::StoreOp>(gate_op.getLoc(), adaptor.getControls()[i], elem_ptr);
            }

            // A little extra trolling for multi-controlled gates with an angle
            // parameter: Need to allocate a buffer to hold both the rotation
            // angle and target qubit pointer. If you don't know why this is
            // needed, then you and I are on the same page, my friend.
            mlir::Type rot_args_struct = mlir::LLVM::LLVMStructType::getLiteral(
                    getContext(),
                    {rewriter.getF64Type(),
                     rewriter.getType<mlir::LLVM::LLVMPointerType>()});
            mlir::Value rot_args_size = sizeofHack(
                    gate_op.getLoc(),
                    rewriter,
                    rot_args_struct,
                    rewriter.getI64Type());
            mlir::Value rot_args_buf = rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(), stubs.tupleCreate, rot_args_size).getResult();
            mlir::Value theta_ptr = rewriter.create<mlir::LLVM::GEPOp>(
                    gate_op.getLoc(),
                    rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                    rot_args_struct,
                    rot_args_buf,
                    std::initializer_list<mlir::LLVM::GEPArg>{0, 0});
            rewriter.create<mlir::LLVM::StoreOp>(
                    gate_op.getLoc(),
                    adaptor.getParam(),
                    theta_ptr);
            mlir::Value target_ptr = rewriter.create<mlir::LLVM::GEPOp>(
                    gate_op.getLoc(),
                    rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                    rot_args_struct,
                    rot_args_buf,
                    std::initializer_list<mlir::LLVM::GEPArg>{0, 1});
            rewriter.create<mlir::LLVM::StoreOp>(
                    gate_op.getLoc(),
                    adaptor.getQubit(),
                    target_ptr);

            // Finally, invoke the multi-control stub!
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    gate_stubs.stub_mc1q,
                    mlir::ValueRange({controls_arr, rot_args_buf}));

            // free() the array
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    stubs.arrayUpdateRc,
                    std::initializer_list<mlir::Value>{
                        controls_arr,
                        rewriter.create<mlir::arith::ConstantOp>(gate_op.getLoc(), rewriter.getI32IntegerAttr(-1)).getResult()});

            // free() the rotation arguments buffer
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    stubs.tupleUpdateRc,
                    std::initializer_list<mlir::Value>{
                        rot_args_buf,
                        rewriter.create<mlir::arith::ConstantOp>(gate_op.getLoc(), rewriter.getI32IntegerAttr(-1)).getResult()});

            // Replace any uses with the qubits we took as input
            llvm::SmallVector<mlir::Value> new_outputs;
            new_outputs.append(adaptor.getControls().begin(), adaptor.getControls().end());
            new_outputs.push_back(adaptor.getQubit());
            rewriter.replaceOp(gate_op, new_outputs);
            return mlir::success();
        }
    }
};

struct Gate2QOpLowering : public mlir::OpConversionPattern<qcirc::Gate2QOp> {
    IntrinsicStubs &stubs;

    Gate2QOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::Gate2QOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::Gate2QOp gate_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        if (gate_op.getControls().empty()) {
            rewriter.create<mlir::LLVM::CallOp>(
                    gate_op.getLoc(),
                    stubs.gates2q[gate_op.getGate()],
                    std::initializer_list<mlir::Value>{adaptor.getLeftQubit(), adaptor.getRightQubit()});
            // Replace any uses with the qubit we took as input
            rewriter.replaceOp(gate_op, {adaptor.getLeftQubit(), adaptor.getRightQubit()});
            return mlir::success();
        } else {
            return mlir::failure();
        }
    }
};

struct QallocOpLowering : public mlir::OpConversionPattern<qcirc::QallocOp> {
    IntrinsicStubs &stubs;

    QallocOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::QallocOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::QallocOp qalloc_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(qalloc_op, stubs.qalloc, mlir::ValueRange());
        return mlir::success();
    }
};

struct QfreeOpLowering : public mlir::OpConversionPattern<qcirc::QfreeOp> {
    IntrinsicStubs &stubs;

    QfreeOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::QfreeOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::QfreeOp qfree_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.create<mlir::LLVM::CallOp>(qfree_op.getLoc(), stubs.reset, adaptor.getQubit());
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(qfree_op, stubs.qfree, adaptor.getQubit());
        return mlir::success();
    }
};

struct QfreeZeroOpLowering : public mlir::OpConversionPattern<qcirc::QfreeZeroOp> {
    IntrinsicStubs &stubs;

    QfreeZeroOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::QfreeZeroOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::QfreeZeroOp qfree_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(qfree_op, stubs.qfree, adaptor.getQubit());
        return mlir::success();
    }
};

struct MeasureOpLowering : public mlir::OpConversionPattern<qcirc::MeasureOp> {
    IntrinsicStubs &stubs;

    MeasureOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::MeasureOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::MeasureOp meas_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Value result = rewriter.create<mlir::LLVM::CallOp>(meas_op.getLoc(), stubs.measure, adaptor.getQubit()).getResult();
        mlir::Value one = rewriter.create<mlir::LLVM::CallOp>(meas_op.getLoc(), stubs.resultGetOne, mlir::ValueRange()).getResult();
        mlir::Value bit = rewriter.create<mlir::LLVM::CallOp>(meas_op.getLoc(), stubs.resultEqual, mlir::ValueRange({result, one})).getResult();
        rewriter.replaceOp(meas_op, {adaptor.getQubit(), bit});
        return mlir::success();
    }
};

struct ArrayPackOpLowering : public mlir::OpConversionPattern<qcirc::ArrayPackOp> {
    IntrinsicStubs &stubs;

    ArrayPackOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::ArrayPackOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::ArrayPackOp pack_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qcirc::ArrayType arr_type = pack_op.getArray().getType();
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(pack_op, "Type converter is null");
        }
        mlir::Type llvm_elem_type = type_conv->convertType(arr_type.getElemType());
        if (!llvm_elem_type) {
            return rewriter.notifyMatchFailure(pack_op, "Cannot convert element type");
        }
        mlir::Location loc = pack_op.getLoc();
        mlir::Value sizeof_elem = sizeofHack(loc, rewriter, llvm_elem_type, rewriter.getI32Type());
        uint64_t dim = arr_type.getDim();
        mlir::Value dim_const = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(dim)).getResult();
        mlir::Value array_ptr = rewriter.create<mlir::LLVM::CallOp>(
                loc, stubs.create1dArray, mlir::ValueRange({sizeof_elem, dim_const})).getResult();

        for (uint64_t i = 0; i < dim; i++) {
            mlir::Value idx = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i)).getResult();
            mlir::Value elem_ptr = rewriter.create<mlir::LLVM::CallOp>(
                    loc, stubs.gep1dArray, mlir::ValueRange({array_ptr, idx})
                ).getResult();
            rewriter.create<mlir::LLVM::StoreOp>(loc, adaptor.getElems()[i], elem_ptr);
        }

        rewriter.replaceOp(pack_op, array_ptr);
        return mlir::success();
    }
};

struct ArrayUnpackOpLowering : public mlir::OpConversionPattern<qcirc::ArrayUnpackOp> {
    IntrinsicStubs &stubs;

    ArrayUnpackOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::ArrayUnpackOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::ArrayUnpackOp unpack_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qcirc::ArrayType arr_type = unpack_op.getArray().getType();
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(unpack_op, "Type converter is null");
        }
        mlir::Type llvm_elem_type = type_conv->convertType(arr_type.getElemType());
        if (!llvm_elem_type) {
            return rewriter.notifyMatchFailure(unpack_op, "Cannot convert element type");
        }
        uint64_t dim = arr_type.getDim();
        mlir::Location loc = unpack_op.getLoc();
        llvm::SmallVector<mlir::Value> elems;
        elems.reserve(dim);

        for (uint64_t i = 0; i < dim; i++) {
            mlir::Value idx = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i)).getResult();
            mlir::Value elem_ptr = rewriter.create<mlir::LLVM::CallOp>(
                    loc, stubs.gep1dArray, mlir::ValueRange({adaptor.getArray(), idx})
                ).getResult();
            elems.push_back(rewriter.create<mlir::LLVM::LoadOp>(loc, llvm_elem_type, elem_ptr).getRes());
        }

        rewriter.replaceOp(unpack_op, elems);
        return mlir::success();
    }
};

struct InitOpLowering : public mlir::OpConversionPattern<qcirc::InitOp> {
    IntrinsicStubs &stubs;

    InitOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::InitOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::InitOp init_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
            init_op, stubs.initialize,
            // Pass NULL because I have no clue what this pointer is
            rewriter.create<mlir::LLVM::ZeroOp>(init_op.getLoc(),
                rewriter.getType<mlir::LLVM::LLVMPointerType>()).getRes());
        return mlir::success();
    }
};

struct QubitIndexOpLowering : public mlir::OpConversionPattern<qcirc::QubitIndexOp> {
    using mlir::OpConversionPattern<qcirc::QubitIndexOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qcirc::QubitIndexOp qidx_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Value qubit_idx_val =
            rewriter.create<mlir::arith::ConstantOp>(
                qidx_op.getLoc(),
                rewriter.getI64IntegerAttr(qidx_op.getIndex())).getResult();
        mlir::Type qubit_ptr_type =
            rewriter.getType<mlir::LLVM::LLVMPointerType>();
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(
            qidx_op, qubit_ptr_type, qubit_idx_val);
        return mlir::success();
    }
};

struct UglyMeasureOpLowering : public mlir::OpConversionPattern<qcirc::UglyMeasureOp> {
    IntrinsicStubs &stubs;

    UglyMeasureOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::UglyMeasureOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::UglyMeasureOp ugly_meas,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        auto loc = ugly_meas.getLoc();
        size_t result_idx = ugly_meas.getResultOffset();
        for (mlir::Value qubit : adaptor.getQubits()) {
            rewriter.create<mlir::LLVM::CallOp>(loc,
                stubs.measureInPlace,
                std::initializer_list<mlir::Value>{
                    qubit,
                    rewriter.create<mlir::LLVM::IntToPtrOp>(loc,
                        rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                        rewriter.create<mlir::arith::ConstantOp>(loc,
                            rewriter.getI64IntegerAttr(result_idx)).getResult()
                    ).getRes()});
            result_idx++;
        }

        rewriter.eraseOp(ugly_meas);
        return mlir::success();
    }
};

struct UglyRecordOpLowering : public mlir::OpConversionPattern<qcirc::UglyRecordOp> {
    IntrinsicStubs &stubs;

    UglyRecordOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::UglyRecordOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::UglyRecordOp ugly_rec,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        auto loc = ugly_rec.getLoc();
        size_t n_results = ugly_rec.getNumResults();
        rewriter.create<mlir::LLVM::CallOp>(loc,
            stubs.tupleRecord,
            std::initializer_list<mlir::Value>{
                rewriter.create<mlir::arith::ConstantOp>(loc,
                    rewriter.getI64IntegerAttr(n_results)).getResult(),
                rewriter.create<mlir::LLVM::GEPOp>(loc,
                    rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                    rewriter.getI8Type(),
                    rewriter.create<mlir::LLVM::AddressOfOp>(loc,
                        rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                        ugly_rec.getUglyLabelAttr()).getRes(),
                    std::initializer_list<mlir::LLVM::GEPArg>{0}).getRes()});

        size_t result_idx = ugly_rec.getResultOffset();
        for (size_t i = 0; i < n_results; i++) {
            rewriter.create<mlir::LLVM::CallOp>(loc,
                stubs.resultRecord,
                std::initializer_list<mlir::Value>{
                    rewriter.create<mlir::LLVM::IntToPtrOp>(loc,
                        rewriter.getType<mlir::LLVM::LLVMPointerType>(),
                        rewriter.create<mlir::arith::ConstantOp>(loc,
                            rewriter.getI64IntegerAttr(result_idx)).getResult()
                    ).getRes(),
                    rewriter.create<mlir::LLVM::ZeroOp>(loc,
                        rewriter.getType<mlir::LLVM::LLVMPointerType>()).getRes()});
            result_idx++;
        }

        rewriter.eraseOp(ugly_rec);
        return mlir::success();
    }
};

struct UglyLabelOpLowering : public mlir::OpConversionPattern<qcirc::UglyLabelOp> {
    using mlir::OpConversionPattern<qcirc::UglyLabelOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qcirc::UglyLabelOp ugly_label,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        createStringConstant(
            rewriter, ugly_label.getLoc(),
            ugly_label.getSymName(), ugly_label.getLabel());
        rewriter.eraseOp(ugly_label);
        return mlir::success();
    }
};

std::string getFuncTableSymName(llvm::StringRef og_metadata_sym_name) {
    return og_metadata_sym_name.str() + "__func_table";
}

std::string getStubSymName(llvm::StringRef og_metadata_sym_name, bool adj) {
    return og_metadata_sym_name.str() + "__" + (adj? "adj" : "fwd") + "_stub";
}

std::string getMemTableSymName(llvm::StringRef og_metadata_sym_name) {
    return og_metadata_sym_name.str() + "__mem_table";
}

std::string getCaptureRefcountSymName(llvm::StringRef og_metadata_sym_name) {
    return og_metadata_sym_name.str() + "__capture_update_reference_count";
}

std::string getCaptureAliasCountSymName(llvm::StringRef og_metadata_sym_name) {
    return og_metadata_sym_name.str() + "__capture_update_alias_count";
}

mlir::Type pointerType(mlir::Builder &builder) {
    return builder.getType<mlir::LLVM::LLVMPointerType>();
}

mlir::Value nullPointer(mlir::OpBuilder &builder, mlir::Location loc) {
    return builder.create<mlir::LLVM::ZeroOp>(
        loc, pointerType(builder)).getRes();
}

mlir::Value funcPtr(mlir::OpBuilder &builder, mlir::Location loc, mlir::FlatSymbolRefAttr sym) {
    return builder.create<mlir::LLVM::AddressOfOp>(
        loc, pointerType(builder), sym).getRes();
}

mlir::Value funcPtr(mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef sym) {
    return funcPtr(
        builder, loc, builder.getAttr<mlir::FlatSymbolRefAttr>(sym));
}

mlir::Value ptrOrNull(mlir::OpBuilder &builder, mlir::Location loc, mlir::FlatSymbolRefAttr stub) {
    if (stub) {
        return funcPtr(builder, loc, stub);
    } else {
        return nullPointer(builder, loc);
    }
}

// What this does is not very complicated (go look at the implementation of
// LLVMTypeConverter::convertFunctionSignature() if you don't believe me), but
// we call it here anyway in case the way it works changes in the future. (For
// example, if they change how multiple results are packed into an LLVM struct)
mlir::LLVM::LLVMFunctionType convertFuncType(
        const mlir::TypeConverter &type_conv, mlir::FunctionType func_ty) {
    const mlir::LLVMTypeConverter &llvm_type_conv =
        static_cast<const mlir::LLVMTypeConverter &>(type_conv);
    // Unused for now, but a required argument
    mlir::LLVMTypeConverter::SignatureConversion sig_conv(
        func_ty.getNumInputs());
    mlir::Type res = llvm_type_conv.convertFunctionSignature(
        func_ty, /*isVariadic=*/false, /*useBarePtrCallConv=*/false, sig_conv);
    return llvm::cast<mlir::LLVM::LLVMFunctionType>(res);
}

mlir::FlatSymbolRefAttr generateStubSkeleton(
        mlir::OpBuilder &builder,
        const mlir::TypeConverter &type_conv,
        mlir::Location loc,
        qcirc::CallableMetadataOp meta,
        qcirc::FuncSpecAttr spec,
        std::function<mlir::Value(mlir::LLVM::LLVMFunctionType,
                                  mlir::ValueRange)> callback) {
    assert(!spec.getNumControls());
    std::string stub_name = getStubSymName(
        meta.getSymName(), spec.getAdjoint());
    llvm::SmallVector<mlir::Type> entry_block_arg_types{
        pointerType(builder), // void *captures
        pointerType(builder), // void *args
        pointerType(builder)  // void *result
    };
    mlir::LLVM::LLVMFunctionType stub_func_type =
        mlir::LLVM::LLVMFunctionType::get(
            /*result=*/builder.getType<mlir::LLVM::LLVMVoidType>(),
            /*params=*/entry_block_arg_types);
    mlir::LLVM::LLVMFuncOp stub_func =
        builder.create<mlir::LLVM::LLVMFuncOp>(
            loc, stub_name, stub_func_type, mlir::LLVM::Linkage::Internal);
    stub_func.setPrivate();

    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        llvm::SmallVector<mlir::Location> entry_block_arg_locs(
            entry_block_arg_types.size(), loc);
        // Sets insert point to end of new block
        mlir::Block *entry_block = builder.createBlock(
            &stub_func.getBody(), {}, entry_block_arg_types, entry_block_arg_locs);
        assert(entry_block->getNumArguments() == entry_block_arg_types.size());
        mlir::Value capture_arg = entry_block->getArgument(0);
        mlir::Value args_arg = entry_block->getArgument(1);
        mlir::Value result_arg = entry_block->getArgument(2);

        llvm::SmallVector<mlir::Value> args;

        // First, populate args with all captures

        // QCirc dialect types
        llvm::SmallVector<mlir::Type> qcirc_capture_types(
            meta.getCaptureTypeRange());
        // ...converted to LLVM dialect types
        llvm::SmallVector<mlir::Type> capture_types;
        mlir::LogicalResult res =
            type_conv.convertTypes(qcirc_capture_types, capture_types);
        if (mlir::failed(res)) {
            assert(0 && "Type conversion for captures failed");
            return nullptr;
        }
        assert(qcirc_capture_types.size() == capture_types.size());

        if (!capture_types.empty()) {
            mlir::LLVM::LLVMStructType capture_tuple =
                mlir::LLVM::LLVMStructType::getLiteral(
                    builder.getContext(), capture_types);

            for (auto [i, capture_type] : llvm::enumerate(capture_types)) {
                mlir::Value gep =
                    builder.create<mlir::LLVM::GEPOp>(
                        loc, pointerType(builder), capture_tuple, capture_arg,
                        std::initializer_list<mlir::LLVM::GEPArg>{
                            0, static_cast<int32_t>(i)},
                        /*inbounds=*/true).getRes();
                mlir::Value loaded =
                    builder.create<mlir::LLVM::LoadOp>(
                        loc, capture_type, gep).getRes();
                args.push_back(loaded);
            }
        }

        // Next, add all actual arguments to args
        mlir::TypeRange spec_inputs = spec.getFunctionType().getInputs();
        assert(spec_inputs.size() >= qcirc_capture_types.size());
        llvm::SmallVector<mlir::Type> qcirc_arg_types(
            spec_inputs.begin() + qcirc_capture_types.size(),
            spec_inputs.end());
        llvm::SmallVector<mlir::Type> arg_types;
        res = type_conv.convertTypes(qcirc_arg_types, arg_types);
        if (mlir::failed(res)) {
            assert(0 && "Type conversion for args failed");
            return nullptr;
        }
        assert(qcirc_arg_types.size() == arg_types.size());

        if (!arg_types.empty()) {
            mlir::LLVM::LLVMStructType arg_tuple =
                mlir::LLVM::LLVMStructType::getLiteral(
                    builder.getContext(), arg_types);

            for (auto [i, arg_type] : llvm::enumerate(arg_types)) {
                mlir::Value gep =
                    builder.create<mlir::LLVM::GEPOp>(
                        loc, pointerType(builder), arg_tuple, args_arg,
                        std::initializer_list<mlir::LLVM::GEPArg>{
                            0, static_cast<int32_t>(i)},
                        /*inbounds=*/true).getRes();
                mlir::Value loaded =
                    builder.create<mlir::LLVM::LoadOp>(
                        loc, arg_type, gep).getRes();
                args.push_back(loaded);
            }
        }

        mlir::LLVM::LLVMFunctionType func_type =
            convertFuncType(type_conv, spec.getFunctionType());
        mlir::Value call_res = callback(func_type, args);

        if (call_res) {
            builder.create<mlir::LLVM::StoreOp>(loc, call_res, result_arg);
        }

        builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange());
    }

    return builder.getAttr<mlir::FlatSymbolRefAttr>(stub_name);
}

mlir::FlatSymbolRefAttr generateStub(
        mlir::OpBuilder &builder,
        const mlir::TypeConverter &type_conv,
        mlir::Location loc,
        qcirc::CallableMetadataOp meta,
        qcirc::FuncSpecAttr spec) {
    return generateStubSkeleton(builder, type_conv, loc, meta, spec, [&](
            mlir::LLVM::LLVMFunctionType func_type,
            mlir::ValueRange args) -> mlir::Value {
        // Now call the actual function!
        mlir::LLVM::CallOp call = builder.create<mlir::LLVM::CallOp>(
            loc, func_type, spec.getSymbol(), args);
        if (llvm::isa<mlir::LLVM::LLVMVoidType>(func_type.getReturnType())) {
            return nullptr;
        } else {
            return call.getResult();
        }
    });
}

mlir::FlatSymbolRefAttr generateControlStub(
        mlir::OpBuilder &builder,
        const mlir::TypeConverter &type_conv,
        IntrinsicStubs &stubs,
        mlir::Location loc,
        qcirc::CallableMetadataOp meta,
        llvm::SmallVectorImpl<qcirc::FuncSpecAttr> &specs) {
    assert(!specs.empty());
    // Why is it kosher to pass this? Well, once all the types are converted to
    // the LLVM dialect (by the type converter), all functions should actually
    // have the same prototype. That is because array<qubit>[N] for any N is
    // lowered to an opaque ptr.
    qcirc::FuncSpecAttr first_spec = specs.front();
    return generateStubSkeleton(builder, type_conv, loc, meta, first_spec, [&](
            mlir::LLVM::LLVMFunctionType func_type,
            mlir::ValueRange args) -> mlir::Value {
        bool is_void =
            llvm::isa<mlir::LLVM::LLVMVoidType>(func_type.getReturnType());
        auto old_insertpt = builder.saveInsertionPoint();
        mlir::Region *body = builder.getInsertionBlock()->getParent();

        mlir::Block *ret_block;
        mlir::Value call_res;
        if (is_void) {
            ret_block = builder.createBlock(body, body->end());
        } else {
            ret_block = builder.createBlock(
                body, body->end(), func_type.getReturnType(), {loc});
            assert(ret_block->getNumArguments() == 1);
            call_res = ret_block->getArgument(0);
        }

        mlir::Block *fallthrough_block = builder.createBlock(ret_block);
        mlir::Value bad_spec_msg_addr =
            builder.create<mlir::LLVM::AddressOfOp>(
                loc, stubs.badSpecMsg).getRes();
        builder.create<mlir::LLVM::CallOp>(loc, stubs.fail, bad_spec_msg_addr);
        builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange());

        llvm::SmallVector<int32_t> case_vals;
        llvm::SmallVector<mlir::Block *> spec_blocks;

        for (qcirc::FuncSpecAttr spec : specs) {
            mlir::FunctionType spec_func_type = spec.getFunctionType();
            assert(spec_func_type.getNumInputs() >= 1
                   && llvm::isa<qcirc::ArrayType>(
                          spec_func_type.getInputs()[spec_func_type.getNumInputs()-1])
                   && llvm::isa<qcirc::QubitType>(
                          llvm::cast<qcirc::ArrayType>(
                              spec_func_type.getInputs()[spec_func_type.getNumInputs()-1]
                          ).getElemType()));
            size_t spec_dim =
                llvm::cast<qcirc::ArrayType>(
                    spec_func_type.getInputs()[spec_func_type.getNumInputs()-1]
                ).getDim();

            case_vals.push_back(spec_dim);
            spec_blocks.push_back(builder.createBlock(ret_block));
            mlir::LLVM::CallOp spec_call =
                builder.create<mlir::LLVM::CallOp>(
                    loc, func_type, spec.getSymbol(), args);
            if (is_void) {
                builder.create<mlir::LLVM::BrOp>(loc, ret_block);
            } else {
                builder.create<mlir::LLVM::BrOp>(loc, spec_call.getResult(), ret_block);
            }
        }

        builder.restoreInsertionPoint(old_insertpt);
        assert(!args.empty());
        mlir::Value qbundle_arg = args[args.size()-1];
        mlir::Value actual_dim = builder.create<mlir::LLVM::CallOp>(
            loc, stubs.size1dArray, qbundle_arg).getResult();
        builder.create<mlir::LLVM::SwitchOp>(
            loc, actual_dim, fallthrough_block, mlir::ValueRange(),
            case_vals, spec_blocks);

        builder.setInsertionPointToEnd(ret_block);
        return call_res;
    });
}

enum class Count {
    Ref,
    Alias
};

// This is a little counterintuitive, but qcirc_capture_type is a Type from
// the qcirc dialect, not a (converted) type in the `llvm' dialect. The
// reason is that `llvm' dialect types lack enough information to know how to
// update the reference count
void updateRefOrAliasCount(
        Count kind, mlir::OpBuilder &builder, IntrinsicStubs &stubs,
        mlir::Location loc, mlir::Type qcirc_capture_type,
        mlir::Value capture_val, mlir::Value delta) {
    if (llvm::isa<qcirc::ArrayType>(qcirc_capture_type)) {
        builder.create<mlir::LLVM::CallOp>(
            loc, kind == Count::Ref ? stubs.arrayUpdateRc
                                    : stubs.arrayUpdateAlias,
            std::initializer_list<mlir::Value>{
                capture_val, delta});
    } else if (llvm::isa<qcirc::CallableType>(qcirc_capture_type)) {
        builder.create<mlir::LLVM::CallOp>(
            loc, kind == Count::Ref ? stubs.callableCaptureUpdateRc
                                    : stubs.callableCaptureUpdateAlias,
            std::initializer_list<mlir::Value>{
                capture_val, delta});
        builder.create<mlir::LLVM::CallOp>(
            loc, kind == Count::Ref ? stubs.callableUpdateRc
                                    : stubs.callableUpdateAlias,
            std::initializer_list<mlir::Value>{
                capture_val, delta});
    } else {
        // TODO: verify somehow that this has no refcount to update?
    }
}

void updateRefOrAliasCount(
        Count kind, mlir::OpBuilder &builder, IntrinsicStubs &stubs,
        mlir::Location loc, mlir::Type qcirc_capture_type,
        mlir::Value capture_val, int32_t delta) {
    mlir::Value const_delta =
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(delta)).getResult();
    updateRefOrAliasCount(kind, builder, stubs, loc, qcirc_capture_type,
                          capture_val, const_delta);
}

void createCaptureRefcountFuncs(
        mlir::OpBuilder &builder,
        const mlir::TypeConverter &type_conv,
        IntrinsicStubs &stubs,
        mlir::Location loc,
        qcirc::CallableMetadataOp meta,
        mlir::FlatSymbolRefAttr &captures_update_refcount_out,
        mlir::FlatSymbolRefAttr &captures_update_alias_count_out) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(meta);

    llvm::SmallVector<mlir::Type> qcirc_capture_types(
        meta.getCaptureTypeRange());
    if (qcirc_capture_types.empty()) {
        captures_update_refcount_out = nullptr;
        captures_update_alias_count_out = nullptr;
        return;
    }

    llvm::SmallVector<mlir::Type> capture_types;
    mlir::LogicalResult res =
        type_conv.convertTypes(qcirc_capture_types, capture_types);
    if (mlir::failed(res)) {
        assert(0 && "Type conversion for captures failed");
        return;
    }
    assert(qcirc_capture_types.size() == capture_types.size());

    std::string refcount_func_name =
        getCaptureRefcountSymName(meta.getSymName());
    std::string alias_count_func_name =
        getCaptureAliasCountSymName(meta.getSymName());

    mlir::LLVM::LLVMStructType capture_tuple_type =
        mlir::LLVM::LLVMStructType::getLiteral(
            builder.getContext(), capture_types);

    mlir::LLVM::LLVMFunctionType refcount_func_ty =
        mlir::LLVM::LLVMFunctionType::get(
            builder.getType<mlir::LLVM::LLVMVoidType>(),
            std::initializer_list<mlir::Type>{
                pointerType(builder), builder.getI32Type()});

    mlir::LLVM::LLVMFuncOp refcount_func =
        builder.create<mlir::LLVM::LLVMFuncOp>(
            loc, refcount_func_name, refcount_func_ty,
            mlir::LLVM::Linkage::Internal);
    refcount_func.setPrivate();
    mlir::LLVM::LLVMFuncOp alias_count_func =
        builder.create<mlir::LLVM::LLVMFuncOp>(
            loc, alias_count_func_name, refcount_func_ty,
            mlir::LLVM::Linkage::Internal);
    alias_count_func.setPrivate();

    // We're about to do something a little crazy: build two similar
    // functions simultaneously: one for updating alias counts, and the other
    // for refcounts

    mlir::Block *ref_entry_block = builder.createBlock(
        &refcount_func.getBody(), {},
        refcount_func_ty.getParams(), {loc, loc});
    mlir::Block *alias_entry_block = builder.createBlock(
        &alias_count_func.getBody(), {},
        refcount_func_ty.getParams(), {loc, loc});

    assert(ref_entry_block->getNumArguments() == 2
           && alias_entry_block->getNumArguments() == 2);
    mlir::Value ref_tuple_arg = ref_entry_block->getArgument(0);
    mlir::Value alias_tuple_arg = alias_entry_block->getArgument(0);
    mlir::Value ref_delta_arg = ref_entry_block->getArgument(1);
    mlir::Value alias_delta_arg = alias_entry_block->getArgument(1);

    for (auto [i, qcirc_capture_type, capture_type] :
            llvm::enumerate(qcirc_capture_types, capture_types)) {
        builder.setInsertionPointToEnd(ref_entry_block);
        mlir::Value ref_gep =
            builder.create<mlir::LLVM::GEPOp>(
                loc, pointerType(builder), capture_tuple_type,
                ref_tuple_arg,
                std::initializer_list<mlir::LLVM::GEPArg>{
                    0, static_cast<int32_t>(i)},
                /*inbounds=*/true).getRes();
        mlir::Value ref_val = builder.create<mlir::LLVM::LoadOp>(
            loc, capture_type, ref_gep).getRes();
        builder.setInsertionPointToEnd(alias_entry_block);
        mlir::Value alias_gep =
            builder.create<mlir::LLVM::GEPOp>(
                loc, pointerType(builder), capture_tuple_type,
                alias_tuple_arg,
                std::initializer_list<mlir::LLVM::GEPArg>{
                    0, static_cast<int32_t>(i)},
                /*inbounds=*/true).getRes();
        mlir::Value alias_val = builder.create<mlir::LLVM::LoadOp>(
            loc, capture_type, alias_gep).getRes();

        builder.setInsertionPointToEnd(ref_entry_block);
        updateRefOrAliasCount(Count::Ref, builder, stubs, loc,
                              qcirc_capture_type,
                              ref_val, ref_delta_arg);
        builder.setInsertionPointToEnd(alias_entry_block);
        updateRefOrAliasCount(Count::Alias, builder, stubs, loc,
                              qcirc_capture_type,
                              alias_val, alias_delta_arg);
    }

    builder.setInsertionPointToEnd(ref_entry_block);
    builder.create<mlir::LLVM::CallOp>(
        loc, stubs.tupleUpdateRc,
        std::initializer_list<mlir::Value>{
            ref_tuple_arg, ref_delta_arg});
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange());

    builder.setInsertionPointToEnd(alias_entry_block);
    builder.create<mlir::LLVM::CallOp>(
        loc, stubs.tupleUpdateAlias,
        std::initializer_list<mlir::Value>{
            alias_tuple_arg, alias_delta_arg});
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange());

    captures_update_refcount_out =
        builder.getAttr<mlir::FlatSymbolRefAttr>(refcount_func_name);
    captures_update_alias_count_out =
        builder.getAttr<mlir::FlatSymbolRefAttr>(alias_count_func_name);
}

struct CallableMetadataOpLowering : public mlir::OpConversionPattern<qcirc::CallableMetadataOp> {
    IntrinsicStubs &stubs;

    CallableMetadataOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableMetadataOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableMetadataOp meta,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = meta.getLoc();
        const mlir::TypeConverter *type_conv = getTypeConverter();
        assert(type_conv
               && "Need a type converter, this isn't pre-kindergarten");
        std::string func_table_name = getFuncTableSymName(meta.getSymName());
        std::string mem_table_name = getMemTableSymName(meta.getSymName());

        // First, build function table

        qcirc::FuncSpecAttr fwd, adj;
        llvm::SmallVector<qcirc::FuncSpecAttr> fwd_ctrl, adj_ctrl;

        for (qcirc::FuncSpecAttr spec : meta.getSpecsRange()) {
            if (spec.getNumControls()) {
                if (spec.getAdjoint()) {
                    adj_ctrl.push_back(spec);
                } else {
                    fwd_ctrl.push_back(spec);
                }
            } else {
                if (spec.getAdjoint()) {
                    assert(!adj);
                    adj = spec;
                } else {
                    assert(!fwd);
                    fwd = spec;
                }
            }
        }

        mlir::FlatSymbolRefAttr fwd_stub, adj_stub,
                                fwd_ctrl_stub, adj_ctrl_stub;
        if (fwd) {
            fwd_stub = generateStub(rewriter, *type_conv, loc, meta, fwd);
        }
        if (adj) {
            adj_stub = generateStub(rewriter, *type_conv, loc, meta, adj);
        }
        if (!fwd_ctrl.empty()) {
            fwd_ctrl_stub = generateControlStub(
                rewriter, *type_conv, stubs, loc, meta, fwd_ctrl);
        }
        if (!adj_ctrl.empty()) {
            adj_ctrl_stub = generateControlStub(
                rewriter, *type_conv, stubs, loc, meta, adj_ctrl);
        }

        mlir::Type func_table_ty =
            rewriter.getType<mlir::LLVM::LLVMArrayType>(
                pointerType(rewriter), 4);
        mlir::LLVM::GlobalOp func_table =
            rewriter.create<mlir::LLVM::GlobalOp>(loc,
                func_table_ty, /*isConstant=*/true,
                mlir::LLVM::Linkage::Internal, func_table_name,
                /*value=*/nullptr);

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            assert(!func_table.getInitializerBlock()
                   && "func_table GlobalOp already has a block, how?");

            // Will set insertion point to end of the block
            rewriter.createBlock(&func_table.getInitializerRegion());

            std::array<mlir::Value, 4> spec_ptrs {
                ptrOrNull(rewriter, loc, fwd_stub),
                ptrOrNull(rewriter, loc, adj_stub),
                ptrOrNull(rewriter, loc, fwd_ctrl_stub),
                ptrOrNull(rewriter, loc, adj_ctrl_stub)
            };

            mlir::Value arr =
                rewriter.create<qcirc::LLVMConstantArrayOp>(loc,
                    func_table_ty, spec_ptrs).getResult();
            rewriter.create<mlir::LLVM::ReturnOp>(loc, arr);
        }

        // Now build memory table (if there are any captures)

        if (!meta.getCaptureTypes().empty()) {
            mlir::Type mem_table_ty =
                rewriter.getType<mlir::LLVM::LLVMArrayType>(
                    pointerType(rewriter), 2);
            mlir::LLVM::GlobalOp mem_table =
                rewriter.create<mlir::LLVM::GlobalOp>(loc,
                    mem_table_ty, /*isConstant=*/true,
                    mlir::LLVM::Linkage::Internal, mem_table_name,
                    /*value=*/nullptr);

            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                assert(!mem_table.getInitializerBlock()
                       && "mem_table GlobalOp already has a block, how?");

                // Will set insertion point to end of the block
                rewriter.createBlock(&mem_table.getInitializerRegion());

                mlir::FlatSymbolRefAttr captures_update_refcount,
                                        captures_update_alias_count;

                createCaptureRefcountFuncs(
                    rewriter, *type_conv, stubs, loc, meta,
                    captures_update_refcount, captures_update_alias_count);

                std::array<mlir::Value, 2> mem_ptrs {
                    ptrOrNull(rewriter, loc, captures_update_refcount),
                    ptrOrNull(rewriter, loc, captures_update_alias_count)
                };

                mlir::Value arr =
                    rewriter.create<qcirc::LLVMConstantArrayOp>(loc,
                        mem_table_ty, mem_ptrs).getResult();
                rewriter.create<mlir::LLVM::ReturnOp>(loc, arr);
            }
        }

        rewriter.eraseOp(meta);
        return mlir::success();
    }
};

struct CallableCreateOpLowering : public mlir::OpConversionPattern<qcirc::CallableCreateOp> {
    IntrinsicStubs &stubs;

    CallableCreateOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableCreateOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableCreateOp create,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = create.getLoc();

        bool no_captures = create.getCaptures().empty();
        mlir::Value func_table =
            funcPtr(rewriter, loc, getFuncTableSymName(create.getMetadata()));

        mlir::Value mem_table, captures;
        if (no_captures) {
            mem_table = nullPointer(rewriter, loc);
            captures = nullPointer(rewriter, loc);
        } else {
            mem_table = funcPtr(rewriter, loc,
                                getMemTableSymName(create.getMetadata()));
            llvm::SmallVector<mlir::Type> capture_types(
                adaptor.getCaptures().getTypes());
            mlir::LLVM::LLVMStructType capture_struct_type =
                mlir::LLVM::LLVMStructType::getLiteral(
                    getContext(), capture_types);
            mlir::Value capture_tuple_size = sizeofHack(
                loc, rewriter, capture_struct_type, rewriter.getI64Type());
            captures =
                rewriter.create<mlir::LLVM::CallOp>(
                    loc, stubs.tupleCreate, capture_tuple_size).getResult();

            assert(adaptor.getCaptures().size()
                   == create.getCaptures().size());
            for (auto [i, capture_val, qcirc_capture_type] :
                    llvm::enumerate(adaptor.getCaptures(),
                                    create.getCaptures().getTypes())) {
                mlir::Value gep =
                    rewriter.create<mlir::LLVM::GEPOp>(
                        loc, pointerType(rewriter), capture_struct_type,
                        captures,
                        std::initializer_list<mlir::LLVM::GEPArg>{
                            0, static_cast<int32_t>(i)},
                        /*inbounds=*/true).getRes();
                rewriter.create<mlir::LLVM::StoreOp>(loc, capture_val, gep);
            }
        }

        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(create,
            stubs.callableCreate, std::initializer_list<mlir::Value>{
                func_table, mem_table, captures}).getResult();
        return mlir::success();
    }
};

struct CallableAdjointOpLowering : public mlir::OpConversionPattern<qcirc::CallableAdjointOp> {
    IntrinsicStubs &stubs;

    CallableAdjointOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableAdjointOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableAdjointOp adj,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = adj.getLoc();
        // At first glance, this may seem blatantly wrong. However, by this
        // point in the code, the copyAndFreeHeapAllocatedObjects()
        // preprocessing as already happened, which will copy the operand to
        // this op automatically (unless it can safely guarantee that isn't
        // necessary). So we are okay to modify the callable in-place.
        rewriter.create<mlir::LLVM::CallOp>(
            loc, stubs.callableAdjoint, adaptor.getCallableIn()).getResult();
        rewriter.replaceOp(adj, adaptor.getCallableIn());
        return mlir::success();
    }
};

struct CallableControlOpLowering : public mlir::OpConversionPattern<qcirc::CallableControlOp> {
    IntrinsicStubs &stubs;

    CallableControlOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableControlOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableControlOp ctrl,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = ctrl.getLoc();
        // copyAndFreeHeapAllocatedObjects() makes this valid. See comment
        // above in CallableAdjointOpLowering.
        rewriter.create<mlir::LLVM::CallOp>(
            loc, stubs.callableControl, adaptor.getCallableIn()).getResult();
        rewriter.replaceOp(ctrl, adaptor.getCallableIn());
        return mlir::success();
    }
};

struct CallableInvokeOpLowering : public mlir::OpConversionPattern<qcirc::CallableInvokeOp> {
    IntrinsicStubs &stubs;

    CallableInvokeOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableInvokeOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableInvokeOp invoke,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = invoke.getLoc();
        const mlir::TypeConverter *type_conv = getTypeConverter();
        assert(type_conv && "Need a type converter");

        mlir::FunctionType qcirc_func_type =
            invoke.getCallable().getType().getFunctionType();
        mlir::LLVM::LLVMFunctionType func_type =
            convertFuncType(*type_conv, qcirc_func_type);

        bool no_inputs = !qcirc_func_type.getNumInputs();
        mlir::Value arg_ptr;
        if (no_inputs) {
            arg_ptr = nullPointer(rewriter, loc);
        } else {
            mlir::LLVM::LLVMStructType arg_struct_type =
                mlir::LLVM::LLVMStructType::getLiteral(
                    getContext(), func_type.getParams());
            mlir::Value arg_size = sizeofHack(
                loc, rewriter, arg_struct_type, rewriter.getI64Type());
            arg_ptr = rewriter.create<mlir::LLVM::CallOp>(
                loc, stubs.tupleCreate, arg_size).getResult();

            assert(adaptor.getCallOperands().size()
                   == func_type.getParams().size());
            for (auto [i, arg_val] :
                    llvm::enumerate(adaptor.getCallOperands())) {
                mlir::Value gep =
                    rewriter.create<mlir::LLVM::GEPOp>(
                        loc, pointerType(rewriter), arg_struct_type,
                        arg_ptr,
                        std::initializer_list<mlir::LLVM::GEPArg>{
                            0, static_cast<int32_t>(i)},
                        /*inbounds=*/true).getRes();
                rewriter.create<mlir::LLVM::StoreOp>(loc, arg_val, gep);
            }
        }

        bool no_results = !qcirc_func_type.getNumResults();
        mlir::Value result_ptr;
        if (no_results) {
            result_ptr = nullPointer(rewriter, loc);
        } else {
            mlir::Type ret_type = func_type.getReturnType();
            mlir::Value ret_size = sizeofHack(
                loc, rewriter, ret_type, rewriter.getI64Type());
            result_ptr = rewriter.create<mlir::LLVM::CallOp>(
                loc, stubs.tupleCreate, ret_size).getResult();
        }

        rewriter.create<mlir::LLVM::CallOp>(
            loc, stubs.callableInvoke,
            std::initializer_list<mlir::Value>{
                adaptor.getCallable(), arg_ptr, result_ptr});

        if (!no_inputs) {
            mlir::Value const_neg_one =
                rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getI32IntegerAttr(-1)).getResult();
            rewriter.create<mlir::LLVM::CallOp>(
                loc, stubs.tupleUpdateRc,
                std::initializer_list<mlir::Value>{
                    arg_ptr, const_neg_one});
        }

        if (no_results) {
            rewriter.eraseOp(invoke);
        } else {
            mlir::Value result =
                rewriter.create<mlir::LLVM::LoadOp>(
                    loc, func_type.getReturnType(), result_ptr).getRes();

            mlir::Value const_neg_one =
                rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getI32IntegerAttr(-1)).getResult();
            rewriter.create<mlir::LLVM::CallOp>(
                loc, stubs.tupleUpdateRc,
                std::initializer_list<mlir::Value>{
                    result_ptr, const_neg_one});

            rewriter.replaceOp(invoke, result);
        }

        return mlir::success();
    }
};

struct CallableCopyOpLowering : public mlir::OpConversionPattern<qcirc::CallableCopyOp> {
    IntrinsicStubs &stubs;

    CallableCopyOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableCopyOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableCopyOp copy,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = copy.getLoc();
        mlir::Value const_true =
            rewriter.create<mlir::arith::ConstantOp>(loc,
                rewriter.getIntegerAttr(rewriter.getI1Type(), 1)).getResult();
        mlir::Value copied = rewriter.create<mlir::LLVM::CallOp>(loc,
            stubs.callableCopy, std::initializer_list<mlir::Value>{
                adaptor.getCallableIn(), /*force_copy=*/const_true
            }).getResult();
        // Whenever we copy the callable, we also need to increment the
        // refcount of its capture tuple
        mlir::Value const_one =
            rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getI32IntegerAttr(1)).getResult();
        rewriter.create<mlir::LLVM::CallOp>(loc,
            stubs.callableCaptureUpdateRc, std::initializer_list<mlir::Value>{
                copied, /*delta=*/const_one});
        rewriter.replaceOp(copy, copied);
        return mlir::success();
    }
};

struct CallableFreeOpLowering : public mlir::OpConversionPattern<qcirc::CallableFreeOp> {
    IntrinsicStubs &stubs;

    CallableFreeOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::CallableFreeOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::CallableFreeOp free,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        updateRefOrAliasCount(Count::Ref, rewriter, stubs, free.getLoc(),
                              free.getCallable().getType(),
                              adaptor.getCallable(), /*delta=*/-1);
        rewriter.eraseOp(free);
        return mlir::success();
    }
};

struct ArrayCopyOpLowering : public mlir::OpConversionPattern<qcirc::ArrayCopyOp> {
    IntrinsicStubs &stubs;

    ArrayCopyOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::ArrayCopyOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::ArrayCopyOp copy,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = copy.getLoc();
        mlir::Value const_true =
            rewriter.create<mlir::arith::ConstantOp>(loc,
                rewriter.getIntegerAttr(rewriter.getI1Type(), 1)).getResult();
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(copy,
            stubs.arrayCopy, std::initializer_list<mlir::Value>{
                adaptor.getArrayIn(), /*force_copy=*/const_true
            }).getResult();
        return mlir::success();
    }
};

struct ArrayFreeOpLowering : public mlir::OpConversionPattern<qcirc::ArrayFreeOp> {
    IntrinsicStubs &stubs;

    ArrayFreeOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context, IntrinsicStubs &stubs)
        : mlir::OpConversionPattern<qcirc::ArrayFreeOp>(typeConverter, context), stubs(stubs) {}

    mlir::LogicalResult matchAndRewrite(qcirc::ArrayFreeOp free,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        updateRefOrAliasCount(Count::Ref, rewriter, stubs, free.getLoc(),
                              free.getArray().getType(), adaptor.getArray(),
                              /*delta=*/-1);
        rewriter.eraseOp(free);
        return mlir::success();
    }
};

struct QCircToQIRConversionPass : public qcirc::QCircToQIRConversionBase<QCircToQIRConversionPass> {
    void runOnOperation() override {
        mlir::ModuleOp module_op = getOperation();
        addExtraLLVMAttrsToFuncs(module_op);
        inlineCalcs(module_op);
        copyAndFreeHeapAllocatedObjects(module_op);

        mlir::LLVMConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp, qcirc::LLVMConstantArrayOp>();

        QCircToQIRTypeConverter typeConverter(&getContext());
        IntrinsicStubsFactory stubFactory(module_op, &getContext());
        IntrinsicStubs stubs = stubFactory.build();

        mlir::RewritePatternSet patterns(&getContext());
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

        // These don't call QIR runtime functions, so stubs aren't needed
        patterns.add<QubitIndexOpLowering,
                     UglyLabelOpLowering>(typeConverter, &getContext());
        // These _do_ call QIR runtime functions, so we need to pass the stubs
        // to their constructor
        patterns.add<Gate1QOpLowering,
                     Gate1Q1POpLowering,
                     Gate2QOpLowering,
                     QallocOpLowering,
                     QfreeOpLowering,
                     QfreeZeroOpLowering,
                     MeasureOpLowering,
                     ArrayPackOpLowering,
                     ArrayUnpackOpLowering,
                     InitOpLowering,
                     UglyMeasureOpLowering,
                     UglyRecordOpLowering,
                     CallableMetadataOpLowering,
                     CallableCreateOpLowering,
                     CallableAdjointOpLowering,
                     CallableControlOpLowering,
                     CallableInvokeOpLowering,
                     CallableCopyOpLowering,
                     CallableFreeOpLowering,
                     ArrayCopyOpLowering,
                     ArrayFreeOpLowering>(typeConverter, &getContext(), stubs);

        if (mlir::failed(mlir::applyFullConversion(module_op, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createQCircToQIRConversionPass() {
    return std::make_unique<QCircToQIRConversionPass>();
}
