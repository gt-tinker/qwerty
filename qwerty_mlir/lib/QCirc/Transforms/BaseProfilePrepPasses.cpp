#include <queue>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Transforms/QCircPasses.h"
#include "PassDetail.h"

// Passes for doing static qubit allocation for the QIR Base Profile. Expects
// code to be fully inlined and for all Qwerty kernels to end in something
// like measure[N] + discard[M]. Measurements from the qubits the programmer
// asked to discard will end up in a tuple named "discarded," and the
// qubits whose measurement results programmers requested will be in a tuple
// named ret.
//
// Right now, we bother with doing all this at the QCirc level because QCirc
// to QIR conversion is pretty methodical and unaware of things like QIR
// profiles. So in this case it's actually easier to set up this structure
// such that the QCirc to QIR conversion can produce llvm-dialect IR in a
// peephole fashion.

namespace {

const llvm::StringRef TAG_RET = "tag_ret";
const llvm::StringRef TAG_DISCARDED = "tag_discarded";

// Add module flags to the LLVM module as required by the QIR base profile:
// https://github.com/qir-alliance/qir-spec/blob/8b3fd47b7b70122a104e24733ef9de911576f7d6/specification/under_development/profiles/Base_Profile.md#module-flags-metadata
// (And also create what will be lowered to global character arrays for the
//  labels of tuples.)
struct BaseProfileModulePrepPass
        : public qcirc::BaseProfileModulePrepBase<BaseProfileModulePrepPass> {
    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();

        mlir::IRRewriter rewriter(&getContext());

        // We want these to be converted to array constants
        rewriter.setInsertionPointToStart(mod.getBody());
        rewriter.create<qcirc::UglyLabelOp>(
            rewriter.getUnknownLoc(),
            rewriter.getStringAttr(TAG_RET),
            rewriter.getStringAttr("ret"));
        rewriter.create<qcirc::UglyLabelOp>(
            rewriter.getUnknownLoc(),
            rewriter.getStringAttr(TAG_DISCARDED),
            rewriter.getStringAttr("discarded"));

        // Just good practice, in my opinion, although nobody is listening to
        // all the events triggered by changing the IR.
        rewriter.modifyOpInPlace(mod, [&]() {
            // There doesn't appear to be a way to set module flags in MLIR. So
            // we will add them as discardable attributes on the module and
            // then let mlir_handle.cpp copy them over into the llvm::Module.
            // TODO: Consider contributing robust support for module flags
            // upstream
            mod->setDiscardableAttr("llvm.flag.qir_major_version",
                                    rewriter.getI32IntegerAttr(1));
            mod->setDiscardableAttr("llvm.flag.qir_minor_version",
                                    rewriter.getI32IntegerAttr(0));
            mod->setDiscardableAttr("llvm.flag.dynamic_qubit_management",
                                    rewriter.getBoolAttr(false));
            mod->setDiscardableAttr("llvm.flag.dynamic_result_management",
                                    rewriter.getBoolAttr(false));
        });
    }
};

// Assume every public func.func op will become a different QIR entry point.
// This pattern will statically allocate qubits and insert the stand-in QIR
// base profile ops as needed to eventually (after conversion to the llvm
// dialect and then translated to LLVM IR) produce QIR.
struct BaseProfileFuncPrepPass
        : public qcirc::BaseProfileFuncPrepBase<BaseProfileFuncPrepPass> {
    void runOnOperation() override {
        mlir::func::FuncOp func = getOperation();

        if (func.isPrivate()) {
            // Not an entry point, nothing to do here
            return;
        }

        mlir::Region &func_body = func.getBody();
        if (!func_body.hasOneBlock()) {
            llvm::errs() << "Expected exactly 1 basic block in the FuncOp\n";
            signalPassFailure();
            return;
        }

        mlir::Block &block = func_body.front();

        if (block.getNumArguments()) {
            llvm::errs() << "Can't handle FuncOps with arguments\n";
            signalPassFailure();
            return;
        }

        size_t n_qubits = 0;
        size_t n_qubits_freed = 0;
        std::vector<bool> measured;
        llvm::DenseMap<mlir::Value, size_t> value_indices;
        // Delay deleting ops to avoid iterator invalidation pain. (size_t)-1
        // is a sentinel that means the op should be deleted entirely
        llvm::SmallVector<std::pair<mlir::Operation *, size_t>> pending_replacement;
        // The bool is true if the qubit should be measured (usually, if it was
        // not measured before being freed), false otherwise
        llvm::SmallVector<std::pair<qcirc::QfreeOp, bool>> qfrees;
        llvm::SmallVector<mlir::Value> freed_ancillas;

        #define THREAD_QUBIT_IDX(gate, arg_name, result_name) \
            auto arg_name ## it = value_indices.find(gate.get ## arg_name()); \
            assert(arg_name ## it != value_indices.end() \
                   && "malformed IR: freeing unknown SSA edge"); \
            size_t arg_name ## _idx = arg_name ## it->getSecond(); \
            value_indices.erase(arg_name ## it); \
            value_indices[gate.get ## result_name()] = arg_name ## _idx;

        #define THREAD_CONTROL_IDXS(gate) \
            for (size_t i = 0; i < gate.getControls().size(); i++) { \
                auto ctrl_it = value_indices.find(gate.getControls()[i]); \
                assert(ctrl_it != value_indices.end() \
                       && "malformed IR: unknown SSA edge"); \
                size_t ctrl_qubit_idx = ctrl_it->getSecond(); \
                value_indices.erase(ctrl_it); \
                value_indices[gate.getControlResults()[i]] = ctrl_qubit_idx; \
            }

        for (mlir::Block::iterator it = block.begin(); it != block.end(); it++) {
            mlir::Operation *op = &*it;

            if (qcirc::QallocOp qalloc = llvm::dyn_cast<qcirc::QallocOp>(op)) {
                if (freed_ancillas.empty()) {
                    size_t qubit_idx = n_qubits++;
                    measured.push_back(false);

                    value_indices[qalloc.getResult()] = qubit_idx;
                    pending_replacement.emplace_back(qalloc, qubit_idx);
                } else {
                    mlir::Value ancilla = freed_ancillas.pop_back_val();
                    // We can safely replace uses with an existing Value
                    // without invalidating the iterator, so do it
                    qalloc.getResult().replaceAllUsesWith(ancilla);
                    pending_replacement.emplace_back(qalloc, (size_t)-1);
                }
            } else if (qcirc::QfreeOp qfree = llvm::dyn_cast<qcirc::QfreeOp>(op)) {
                auto it = value_indices.find(qfree.getQubit());
                assert(it != value_indices.end()
                       && "malformed IR: freeing unknown SSA edge");
                size_t qubit_idx = it->getSecond();
                n_qubits_freed++;
                value_indices.erase(it);
                qfrees.emplace_back(qfree, !measured[qubit_idx]);
            } else if (qcirc::QfreeZeroOp qfreez = llvm::dyn_cast<qcirc::QfreeZeroOp>(op)) {
                freed_ancillas.push_back(qfreez.getQubit());
                // Don't measure this qubit (it maybe used later). Just erase it
                qfrees.emplace_back(qfreez, false);
            } else if (qcirc::MeasureOp meas = llvm::dyn_cast<qcirc::MeasureOp>(op)) {
                THREAD_QUBIT_IDX(meas, Qubit, QubitResult)
                if (measured[Qubit_idx]) {
                    llvm::errs() << "Measuring a qubit twice is not "
                                    "supported\n";
                    signalPassFailure();
                    return;
                }
                measured[Qubit_idx] = true;
            } else if (qcirc::Gate1QOp gate1q = llvm::dyn_cast<qcirc::Gate1QOp>(op)) {
                THREAD_QUBIT_IDX(gate1q, Qubit, Result)
                THREAD_CONTROL_IDXS(gate1q)
            } else if (qcirc::Gate1Q1POp gate1q1p = llvm::dyn_cast<qcirc::Gate1Q1POp>(op)) {
                THREAD_QUBIT_IDX(gate1q1p, Qubit, Result)
                THREAD_CONTROL_IDXS(gate1q1p)
            } else if (qcirc::Gate1Q3POp gate1q3p = llvm::dyn_cast<qcirc::Gate1Q3POp>(op)) {
                THREAD_QUBIT_IDX(gate1q3p, Qubit, Result)
                THREAD_CONTROL_IDXS(gate1q3p)
            } else if (qcirc::Gate2QOp gate2q = llvm::dyn_cast<qcirc::Gate2QOp>(op)) {
                THREAD_QUBIT_IDX(gate2q, LeftQubit, LeftResult)
                THREAD_QUBIT_IDX(gate2q, RightQubit, RightResult)
                THREAD_CONTROL_IDXS(gate2q)
            } else {
                for (mlir::Type type : op->getOperandTypes()) {
                    if (llvm::isa<qcirc::QubitType>(type)) {
                        llvm::errs() << "Unknown operation " << op->getName()
                                     << " with qubit input\n";
                        signalPassFailure();
                        return;
                    }
                }
                for (mlir::Type type : op->getResultTypes()) {
                    if (llvm::isa<qcirc::QubitType>(type)) {
                        llvm::errs() << "Unknown operation " << op->getName()
                                     << " with qubit output\n";
                        signalPassFailure();
                        return;
                    }
                }
            }
        }

        #undef THREAD_QUBIT_IDX
        #undef THREAD_CONTROL_IDXS

        if (n_qubits_freed + freed_ancillas.size() != n_qubits) {
            llvm::errs() << "Not all qubits freed\n";
            signalPassFailure();
            return;
        }

        // Now that we have no iterator to blow up, we can erase
        // qalloc/qfree(z) ops

        mlir::IRRewriter rewriter(&getContext());

        // Need to initialize the runtime
        rewriter.setInsertionPoint(&block.front());
        rewriter.create<qcirc::InitOp>(rewriter.getUnknownLoc());

        for (auto [op, qubit_idx] : pending_replacement) {
            if (qubit_idx == (size_t)-1) {
                rewriter.eraseOp(op);
            } else {
                rewriter.setInsertionPoint(op);
                rewriter.replaceOpWithNewOp<qcirc::QubitIndexOp>(
                    op, qubit_idx);
            }
        }
        llvm::SmallVector<mlir::Value> qubits_to_discard;
        for (auto &[qfree, should_measure] : qfrees) {
            if (should_measure) {
                qubits_to_discard.push_back(qfree.getQubit());
            }
            rewriter.eraseOp(qfree);
        }
        // Measure all qubits that were qfreez'd and not reused
        qubits_to_discard.append(freed_ancillas);

        mlir::func::ReturnOp ret;
        if (!(ret = llvm::dyn_cast<mlir::func::ReturnOp>(
                block.getTerminator()))) {
            llvm::errs() << "Terminator is not a ReturnOp. How?\n";
            signalPassFailure();
            return;
        }

        qcirc::ArrayPackOp bitpack;
        if (ret.getNumOperands() != 1
                || !(bitpack =
                     ret.getOperand(0)
                        .getDefiningOp<qcirc::ArrayPackOp>())
                || bitpack.getArray().getType().getElemType()
                   != rewriter.getI1Type()
                // So we know we can delete it safely
                || !bitpack.getArray().hasOneUse()) {
            llvm::errs() << "Not returning a bitpack with 1 use, no hope\n";
            signalPassFailure();
            return;
        }

        llvm::SmallVector<qcirc::MeasureOp> measures_to_kill;
        llvm::SmallVector<mlir::Value> qubits_to_measure;
        for (mlir::Value bit : bitpack.getElems()) {
            qcirc::MeasureOp meas;
            if (!(meas = bit.getDefiningOp<qcirc::MeasureOp>())) {
                llvm::errs() << "Can't return a non-measurement";
                signalPassFailure();
                return;
            }
            qubits_to_measure.push_back(meas.getQubit());
            measures_to_kill.push_back(meas);
        }

        if (qubits_to_discard.size() + qubits_to_measure.size() != n_qubits) {
            llvm::errs() << "Unknown IR structure";
            signalPassFailure();
            return;
        }

        rewriter.setInsertionPoint(bitpack);
        rewriter.create<qcirc::UglyMeasureOp>(bitpack.getLoc(),
            0, qubits_to_measure);
        rewriter.create<qcirc::UglyMeasureOp>(rewriter.getUnknownLoc(),
            qubits_to_measure.size(), qubits_to_discard);

        rewriter.create<qcirc::UglyRecordOp>(rewriter.getUnknownLoc(),
            TAG_DISCARDED, qubits_to_measure.size(), qubits_to_discard.size());
        rewriter.create<qcirc::UglyRecordOp>(bitpack.getLoc(),
            TAG_RET, 0, qubits_to_measure.size());

        mlir::Value success = rewriter.create<mlir::arith::ConstantOp>(
            rewriter.getUnknownLoc(),
            rewriter.getI64IntegerAttr(0));
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(ret, success);

        // Just good practice, in my opinion, although nobody is listening to
        // all the events triggered by changing the IR
        rewriter.modifyOpInPlace(func, [&]() {
            func.setFunctionType(
                rewriter.getFunctionType(mlir::TypeRange(),
                                         rewriter.getI64Type()));
            // TODO: Apparently this is very frowned upon to use:
            //       > WARNING: this feature MUST NOT be used for any real workload.
            //       per https://mlir.llvm.org/docs/Dialects/LLVM/#attribute-pass-through
            func->setDiscardableAttr("passthrough",
                rewriter.getArrayAttr({
                    rewriter.getStringAttr("entry_point"),
                    rewriter.getArrayAttr({
                        rewriter.getStringAttr("qir_profiles"),
                        rewriter.getStringAttr("base_profile")
                    }),
                    rewriter.getArrayAttr({
                        rewriter.getStringAttr("output_labeling_schema"),
                        rewriter.getStringAttr("qwerty_v1")
                    }),
                    rewriter.getArrayAttr({
                        rewriter.getStringAttr("required_num_qubits"),
                        rewriter.getStringAttr(std::to_string(n_qubits))
                    }),
                    rewriter.getArrayAttr({
                        rewriter.getStringAttr("required_num_results"),
                        rewriter.getStringAttr(std::to_string(n_qubits))
                    })
                }));
        });

        rewriter.eraseOp(bitpack);
        for (qcirc::MeasureOp meas : measures_to_kill) {
            rewriter.eraseOp(meas);
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qcirc::createBaseProfileModulePrepPass() {
    return std::make_unique<BaseProfileModulePrepPass>();
}

std::unique_ptr<mlir::Pass> qcirc::createBaseProfileFuncPrepPass() {
    return std::make_unique<BaseProfileFuncPrepPass>();
}
