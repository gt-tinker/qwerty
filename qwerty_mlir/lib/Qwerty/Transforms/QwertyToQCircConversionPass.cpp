// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include <deque>
#include <unordered_map>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

#include "Eigen/SparseCore"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircTypes.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"
#include "Qwerty/Analysis/FuncSpecAnalysis.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Utils/QwertyUtils.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "PassDetail.h"

#include "tweedledum.hpp"

// This pass converts Qwerty constructs to quantum circuits. Most of Sections
// 6.1-6.3 of the CGO paper happen in this file. Your best bet for navigating
// this file is to work backwards from the pass itself (at the bottom of this
// file). With a few exceptions, there is a dialect conversion pattern for each
// op in the Qwerty dialect, so you can hone in on the Qwerty construct whose
// lowering you are interested in.

namespace {

struct FuncSpec {
    bool is_adj;
    size_t n_controls;

    FuncSpec() : is_adj(false), n_controls(0) {}

    FuncSpec(bool is_adj, size_t n_controls)
            : is_adj(is_adj), n_controls(n_controls) {}

    FuncSpec(qwerty::CallOp call)
            : is_adj(call.getAdj()),
              n_controls(call.getPred()? call.getPredAttr().getDim() : 0) {}

    FuncSpec(qwerty::FuncSpec spec)
            : is_adj(spec.is_adj), n_controls(spec.n_controls) {}

    bool operator==(const FuncSpec &other) const {
        return other.is_adj == is_adj
               && other.n_controls == n_controls;
    }

    bool operator<(const FuncSpec &other) const {
        return static_cast<int>(other.is_adj) < static_cast<int>(is_adj)
               || (other.is_adj == is_adj
                   && other.n_controls < n_controls);
    }

    bool isCallableFromPython() const {
        return !is_adj && !n_controls;
    }

    std::string getSymName(llvm::StringRef og_func_name) const {
        if (isCallableFromPython()) {
            // Make it easier for the ExecutionEngine to call this by leaving the
            // original name
            return og_func_name.str();
        } else {
            return (og_func_name + "__"
                    + (is_adj? "adj" : "fwd")
                    + (n_controls? "__ctrl" + std::to_string(n_controls) : "")
                   ).str();
        }
    }

    static std::string getMetadataSymName(llvm::StringRef og_func_name) {
        return (og_func_name + "__metadata").str();
    }

    qcirc::FuncSpecAttr getAttr(mlir::MLIRContext *ctx,
                                 llvm::StringRef og_func_name,
                                 mlir::FunctionType func_type) const {
        return qcirc::FuncSpecAttr::get(
            ctx, is_adj, n_controls,
            mlir::FlatSymbolRefAttr::get(ctx, getSymName(og_func_name)),
            func_type);
    }
};

struct FuncUsage {
    llvm::SmallSet<FuncSpec, 4> specs_needed;
    bool taken_as_const;

    FuncUsage() : specs_needed(), taken_as_const(false) {}
};

using FuncUsages = std::unordered_map<std::string, FuncUsage>;

// Find the function specializations that we need to synthesize (Section 6.2 of
// the CGO paper)
mlir::LogicalResult findFuncUsages(mlir::ModuleOp module_op, FuncUsages &usages) {
    // Functions that have been seen in func_const ops, indexed by type
    llvm::DenseMap<mlir::FunctionType,
                   llvm::SmallVector<llvm::StringRef>> func_types;
    // Reversible functions of the form (@symbol_name, n_qubits) who are seen
    // in func_const ops
    llvm::SmallVector<std::pair<llvm::StringRef, size_t>> rev_funcs;

    for (qwerty::FuncOp func : module_op.getBodyRegion()
                                        .getOps<qwerty::FuncOp>()) {
        std::string name = func.getSymName().str();
        bool taken_as_const = false;

        if (func.isPublic()) {
            // This could be called from Python
            assert(!usages.count(name)
                   && "Duplicate symbol names?");
            usages[name].specs_needed.insert(
                FuncSpec(/*is_adj=*/false, /*n_controls=*/0));
        }

        auto func_sym_uses_opt =
            mlir::SymbolTable::getSymbolUses(func, module_op);
        assert(func_sym_uses_opt.has_value()
               && "Missing uses from symbol table, bug!");
        for (const mlir::SymbolTable::SymbolUse &sym_use
                : func_sym_uses_opt.value()) {
            if (llvm::isa<qwerty::FuncConstOp>(sym_use.getUser())) {
                taken_as_const = true;
                break;
            }
        }
        usages[name].taken_as_const = taken_as_const;

        if (taken_as_const) {
            qwerty::FunctionType func_type = func.getQwertyFuncType();
            mlir::FunctionType inner_func_type = func_type.getFunctionType();
            func_types[inner_func_type].push_back(func.getSymName());

            if (func_type.getReversible()) {
                assert(inner_func_type.getNumInputs() == 1
                       && inner_func_type.getNumResults() == 1
                       && "Reversible qwerty.func should have 1 input and output: "
                          "a qbundle");

                qwerty::QBundleType qbundle_ty =
                    llvm::cast<qwerty::QBundleType>(
                        inner_func_type.getInputs()[0]);
                rev_funcs.emplace_back(func.getSymName(), qbundle_ty.getDim());
            }
        }
    }

    module_op->walk([&](qwerty::CallOp call) {
        std::string callee_name = call.getCallee().str();
        usages[callee_name].specs_needed.insert(FuncSpec(call));
    });

    mlir::DataFlowSolver solver;
    // TODO: Why does this break without these two?
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<qwerty::FuncSpecAnalysis>();
    if (mlir::failed(solver.initializeAndRun(module_op))) {
        return mlir::failure();
    }

    bool adjointed_funcs_bottom = false, predded_funcs_bottom = false;
    llvm::SmallSet<llvm::StringRef, 4> adjointed_funcs, predded_funcs;

    module_op->walk([&](qwerty::FuncAdjointOp func_adj) {
        mlir::Value callee = func_adj.getCallee();
        const qwerty::FuncSpecLattice *lattice =
            solver.lookupState<qwerty::FuncSpecLattice>(callee);
        assert(lattice && "Func spec analysis missing for func_adj. "
                          "Analysis bug?");
        const qwerty::FuncSpecSet &lattice_val = lattice->getValue();
        if (lattice_val.is_bottom) {
            adjointed_funcs_bottom = true;
            return mlir::WalkResult::interrupt();
        } else {
            for (const qwerty::FuncSpec &spec : lattice_val.specs) {
                if (!spec.is_adj) {
                    const llvm::StringRef &callee_name = spec.sym;
                    adjointed_funcs.insert(callee_name);
                }
            }
            return mlir::WalkResult::advance();
        }
    });

    module_op->walk([&](qwerty::FuncPredOp func_pred) {
        mlir::Value callee = func_pred.getCallee();
        const qwerty::FuncSpecLattice *lattice =
            solver.lookupState<qwerty::FuncSpecLattice>(callee);
        assert(lattice && "Func spec analysis missing for func_pred. "
                          "Analysis bug?");
        const qwerty::FuncSpecSet &lattice_val = lattice->getValue();
        if (lattice_val.is_bottom) {
            predded_funcs_bottom = true;
            return mlir::WalkResult::interrupt();
        } else {
            for (const qwerty::FuncSpec &spec : lattice_val.specs) {
                const llvm::StringRef &callee_name = spec.sym;
                predded_funcs.insert(callee_name);
            }
            return mlir::WalkResult::advance();
        }
    });

    module_op->walk([&](qwerty::CallIndirectOp calli) {
        mlir::Value callee = calli.getCallee();
        const qwerty::FuncSpecLattice *lattice =
            solver.lookupState<qwerty::FuncSpecLattice>(callee);
        assert(lattice && "Func spec analysis missing for call_indirect. "
                          "Analysis bug?");
        const qwerty::FuncSpecSet &lattice_val = lattice->getValue();
        if (!lattice_val.is_bottom) {
            for (const qwerty::FuncSpec &spec : lattice_val.specs) {
                const std::string &callee_name = spec.sym;
                usages[callee_name].specs_needed.insert(FuncSpec(spec));
            }
        } else {
            // Well, this is some unfortunate imprecision
            mlir::FunctionType callee_func_type =
                mlir::FunctionType::get(
                    module_op.getContext(),
                    calli.getCallOperands().getTypes(),
                    calli.getResults().getTypes());
            if (func_types.count(callee_func_type)) {
                for (llvm::StringRef func_name :
                        func_types.at(callee_func_type)) {
                    std::string func_name_str = func_name.str();
                    usages[func_name_str].specs_needed.insert(
                        FuncSpec(/*is_adj=*/false, /*n_controls=*/0));
                }
            }

            qwerty::QBundleType qbundle_in, qbundle_out;
            if (calli.getCallOperands().size() == 1
                    && calli.getResults().size() == 1
                    && (qbundle_in = llvm::dyn_cast<qwerty::QBundleType>(
                            calli.getCallOperands()[0].getType()))
                    && (qbundle_out = llvm::dyn_cast<qwerty::QBundleType>(
                            calli.getResults()[0].getType()))
                    && qbundle_in == qbundle_out) {
                size_t n_call_qubits = qbundle_in.getDim();

                for (auto [name, n_callee_qubits] : rev_funcs) {
                    if (n_call_qubits < n_callee_qubits) {
                        // Impossible to be the callee. You can't un-predicate!
                        continue;
                    }

                    std::string maybe_callee_name = name.str();
                    if (n_call_qubits == n_callee_qubits) {
                        if (adjointed_funcs_bottom || adjointed_funcs.count(maybe_callee_name)) {
                            // The code above already added the forward version,
                            // now we should add the adjoint
                            usages[maybe_callee_name].specs_needed.insert(
                                FuncSpec(/*is_adj=*/true, /*n_controls=*/0));
                        }
                    } else if (n_call_qubits > n_callee_qubits) {
                        size_t n_pred = n_call_qubits - n_callee_qubits;
                        if (predded_funcs_bottom || predded_funcs.count(maybe_callee_name)) {
                            usages[maybe_callee_name].specs_needed.insert(
                                FuncSpec(/*is_adj=*/false, n_pred));
                            if (adjointed_funcs_bottom || adjointed_funcs.count(maybe_callee_name)) {
                                usages[maybe_callee_name].specs_needed.insert(
                                    FuncSpec(/*is_adj=*/true, n_pred));
                            }
                        }
                    }
                }
            }
        }
    });

    DEBUG_WITH_TYPE("find-func-usages", {
        llvm::errs() << "final func usages\n";
        for (auto &[sym, sym_usage] : usages) {
            llvm::errs() << sym << ": [";
            llvm::interleaveComma(sym_usage.specs_needed, llvm::errs(),
                [](auto &spec) {
                    llvm::errs() << "(" << (spec.is_adj? "adj" : "fwd") << ","
                                 << spec.n_controls << ")";
                });
            llvm::errs() << "]\n";
        }
    });

    return mlir::success();
}

mlir::FunctionType convertFuncType(qwerty::FuncOp func,
                                   const mlir::TypeConverter &type_conv) {
    mlir::MLIRContext *ctx = func.getContext();
    mlir::Type func_type_with_captures =
            mlir::FunctionType::get(
                ctx,
                func.getBody().getArgumentTypes(),
                func.getQwertyFuncType()
                    .getFunctionType()
                    .getResults());
    mlir::FunctionType new_func_type =
        llvm::dyn_cast<mlir::FunctionType>(
            type_conv.convertType(func_type_with_captures));
    assert(new_func_type);
    return new_func_type;
}

// Generate the function specializations that we need to synthesize (Section
// 6.2 of the CGO paper).
// Originally, this code was inside FuncOpLowering below, but unfortunately,
// the ConversionPatternRewriter appears to be too primitive to handle the
// region/block manipulation done below. Thus, we run this code before dialect
// conversion begins, producing some qwerty.func ops that get lowered 1:1 by
// FuncOpLowering to func.func ops.
mlir::LogicalResult generateFuncSpecs(
        mlir::ModuleOp module_op, FuncUsages &usages,
        const mlir::TypeConverter &type_conv) {
    mlir::IRRewriter rewriter(module_op.getContext());

    llvm::SmallVector<qwerty::FuncOp> func_queue(
        module_op.getBodyRegion().getOps<qwerty::FuncOp>());

    for (qwerty::FuncOp func : func_queue) {
        if (usages[func.getSymName().str()].specs_needed.empty()) {
            // Dead code
            rewriter.eraseOp(func);
            continue;
        }

        mlir::Location loc = func.getLoc();
        // Can't call getCaptureTypes() after we inline (steal) the region, so
        // speculatively do it here (since we might not need it below)
        llvm::SmallVector<mlir::Type> capture_types;
        func.getCaptureTypes(capture_types);

        llvm::SmallVector<mlir::Attribute> spec_attrs;

        auto &usage = usages[func.getSymName().str()];
        for (auto [i, spec] : llvm::enumerate(usage.specs_needed)) {
            bool last = i+1 == usage.specs_needed.size();
            qwerty::QBundleType bigger_qbundle_ty;
            qwerty::FunctionType new_func_type;
            if (!spec.n_controls) {
                new_func_type = func.getQwertyFuncType();
            } else {
                mlir::FunctionType inner_func_ty =
                    func.getQwertyFuncType().getFunctionType();
                llvm::SmallVector<mlir::Type> new_inputs(
                    inner_func_ty.getInputs());
                llvm::SmallVector<mlir::Type> new_results(
                    inner_func_ty.getResults());
                assert(func.getQwertyFuncType().getReversible()
                       && !new_inputs.empty()
                       && !new_results.empty()
                       && new_inputs[new_inputs.size()-1] == new_results[new_results.size()-1]
                       && llvm::isa<qwerty::QBundleType>(new_inputs[new_inputs.size()-1])
                       && "Invariant of reversible functions violated");
                qwerty::QBundleType qbundle_ty =
                    llvm::cast<qwerty::QBundleType>(
                        new_inputs[new_inputs.size()-1]);
                bigger_qbundle_ty =
                    rewriter.getType<qwerty::QBundleType>(
                        qbundle_ty.getDim() + spec.n_controls);
                new_inputs[new_inputs.size()-1] = bigger_qbundle_ty;
                new_results[new_results.size()-1] = bigger_qbundle_ty;

                new_func_type = rewriter.getType<qwerty::FunctionType>(
                    rewriter.getType<mlir::FunctionType>(
                        new_inputs, new_results),
                    /*reversible=*/true);
            }

            std::string new_func_name =
                spec.getSymName(func.getSymName());
            rewriter.setInsertionPoint(func);
            qwerty::FuncOp new_func =
                rewriter.create<qwerty::FuncOp>(loc,
                    new_func_name, new_func_type);
            // Teensy tiny optimization: steal the body of the function for the
            // last specialization instead of needlessly cloning it
            if (last) {
                rewriter.inlineRegionBefore(func.getBody(),
                                            new_func.getBody(),
                                            new_func.getBody().end());
            } else {
                rewriter.cloneRegionBefore(func.getBody(),
                                           new_func.getBody(),
                                           new_func.getBody().end());
            }

            if (func.isPrivate() || !spec.isCallableFromPython()) {
                new_func.setPrivate();
            }

            assert(new_func.getBody().hasOneBlock());
            mlir::Block *block = &new_func.getBody().front();

            if (spec.n_controls) {
                assert(block->getNumArguments());
                // We need to change the type of the last block argument.
                // An easy way to do this is creating a new block with the
                // right arguments and merging it with this block. Thankfully
                // this is O(K) if there are K arguments (that is, it will not
                // reallocate the instructions in the block)
                mlir::TypeRange old_arg_types = block->getArgumentTypes();
                llvm::SmallVector<mlir::Type> new_block_arg_types(
                    old_arg_types.begin(),
                    old_arg_types.begin() + (block->getNumArguments()-1));
                new_block_arg_types.push_back(bigger_qbundle_ty);
                llvm::SmallVector<mlir::Location> new_block_arg_locs(
                    new_block_arg_types.size(), loc);

                // Sets insertion point to beginning of block
                mlir::Block *new_block = rewriter.createBlock(
                    block, new_block_arg_types, new_block_arg_locs);
                assert(new_block->getNumArguments()
                       == block->getNumArguments());
                mlir::Value new_qbundle_arg =
                    new_block->getArgument(new_block->getNumArguments()-1);
                mlir::ValueRange unpacked =
                    rewriter.create<qwerty::QBundleUnpackOp>(
                        loc, new_qbundle_arg).getQubits();
                size_t pred_dim = spec.n_controls;
                mlir::Value pred_qbundle =
                    rewriter.create<qwerty::QBundlePackOp>(
                        loc,
                        llvm::iterator_range(unpacked.begin(),
                                             unpacked.begin()+pred_dim)
                    ).getQbundle();
                qwerty::QBundlePackOp arg_pack_op =
                    rewriter.create<qwerty::QBundlePackOp>(
                        loc,
                        llvm::iterator_range(unpacked.begin()+pred_dim,
                                             unpacked.end()));
                mlir::Value arg_qbundle = arg_pack_op.getQbundle();

                llvm::SmallVector<mlir::Value> old_arg_vals(
                    new_block->args_begin(),
                    new_block->args_begin() + (block->getNumArguments()-1));
                old_arg_vals.push_back(arg_qbundle);
                rewriter.mergeBlocks(block, new_block, old_arg_vals);
                block = new_block;

                // Sneaky trick: we want the predication code to imagine that
                // the last block argument is actually the qbundle of argument
                // (non-pred) qubits that we just packed. This is exactly what
                // the block_args argument to predicateBlockInPlaceFixTerm() is
                // for, so use it
                llvm::SmallVector<mlir::Value> pretend_block_args(
                    block->getArguments());
                assert(!pretend_block_args.empty());
                pretend_block_args[pretend_block_args.size()-1] = arg_qbundle;
                // Also, the predication code should do its qubit index
                // analysis starting right after the qbpack for the
                // aforementioned arg qbundle, since we are imagining that is
                // the start of the bundle
                mlir::Operation *start_at = arg_pack_op->getNextNode();

                qwerty::BasisAttr pred_basis =
                    qwerty::BasisAttr::getAllOnesBasis(
                        func.getContext(), pred_dim);
                if (mlir::failed(
                        qwerty::predicateBlockInPlaceFixTerm<qwerty::ReturnOp>(
                            pred_basis, pred_qbundle, rewriter,
                            *block, start_at, pretend_block_args))) {
                    return func->emitOpError("Predication failed");
                }
            }

            if (spec.is_adj) {
                if (mlir::failed(
                        qcirc::takeAdjointOfBlockInPlace<qwerty::ReturnOp>(
                            rewriter, *block, loc))) {
                    return func->emitOpError("Taking adjoint failed");
                }
            }

            mlir::FunctionType qcirc_func_type =
                convertFuncType(new_func, type_conv);
            spec_attrs.push_back(spec.getAttr(
                func->getContext(), func.getSymName(), qcirc_func_type));
        }

        rewriter.setInsertionPoint(func);

        if (usage.taken_as_const) {
            llvm::SmallVector<mlir::Type> qcirc_capture_types;
            if (mlir::failed(type_conv.convertTypes(
                    capture_types, qcirc_capture_types))) {
                return func->emitOpError("Cannot convert capture types");
            }

            qcirc::CallableMetadataOp meta =
                rewriter.create<qcirc::CallableMetadataOp>(loc,
                    FuncSpec::getMetadataSymName(func.getSymName()),
                    rewriter.getTypeArrayAttr(qcirc_capture_types),
                    rewriter.getArrayAttr(spec_attrs));
            meta.setPrivate();
        }

        rewriter.eraseOp(func);
    }

    return mlir::success();
}

class QwertyToQCircTypeConverter : public mlir::TypeConverter {
public:
    QwertyToQCircTypeConverter(mlir::MLIRContext *ctx) {
        // Fallback to letting stuff through (MLIR checks the list of
        // conversions in reverse order)
        addConversion([=](mlir::Type ty) { return ty; });

        addConversion([=](qwerty::QBundleType qbundle) {
            return qcirc::ArrayType::get(
                ctx, qcirc::QubitType::get(ctx), qbundle.getDim());
        });
        addConversion([=](qwerty::BitBundleType bit_bundle) {
            return qcirc::ArrayType::get(
                ctx, mlir::IntegerType::get(ctx, 1), bit_bundle.getDim());
        });
        addConversion([=](mlir::FunctionType func_ty) -> std::optional<mlir::Type> {
            llvm::SmallVector<mlir::Type> inputs, results;
            inputs.reserve(func_ty.getNumInputs());
            results.reserve(func_ty.getNumResults());
            if (mlir::failed(convertTypes(func_ty.getInputs(), inputs))
                    || mlir::failed(convertTypes(func_ty.getResults(), results))) {
                return std::nullopt;
            }
            return mlir::FunctionType::get(ctx, inputs, results);
        });
        addConversion([=](qwerty::FunctionType func_ty) -> std::optional<mlir::Type> {
            mlir::FunctionType inner_func_ty = func_ty.getFunctionType();
            if (mlir::FunctionType converted_func_ty =
                    convertType<mlir::FunctionType>(inner_func_ty)) {
                return qcirc::CallableType::get(ctx, converted_func_ty);
            } else {
                return std::nullopt;
            }
        });
    }
};

mlir::Value wrapStationaryFloatOps(mlir::OpBuilder &builder,
                                   mlir::Location loc,
                                   mlir::ValueRange args,
                                   std::function<mlir::Value(mlir::ValueRange)> build_body) {
    qcirc::CalcOp calc = builder.create<qcirc::CalcOp>(loc, builder.getF64Type(), args);
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        // Sets insertion point to end of this block
        llvm::SmallVector<mlir::Location> arg_locs(args.size(), loc);
        mlir::Block *calc_block =builder.createBlock(&calc.getRegion(), {}, args.getTypes(), arg_locs);
        assert(calc_block->getNumArguments() == args.size());
        mlir::Value body_ret = build_body(calc_block->getArguments());
        builder.create<qcirc::CalcYieldOp>(loc, body_ret);
    }
    mlir::ValueRange calc_results = calc.getResults();
    assert(calc_results.size() == 1);
    return calc_results[0];
}

// We have to manually convert the result type of scf.if ops
struct SCFIfOpTypeFix : public mlir::OpConversionPattern<mlir::scf::IfOp> {
    using mlir::OpConversionPattern<mlir::scf::IfOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::scf::IfOp if_op,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(if_op, "Need a type converter");
        }
        llvm::SmallVector<mlir::Type> new_result_types;
        if (mlir::failed(type_conv->convertTypes(
                if_op->getResultTypes(), new_result_types))) {
            return rewriter.notifyMatchFailure(if_op, "Could not convert result types");
        }

        mlir::Location loc = if_op.getLoc();
        mlir::scf::IfOp new_if_op =
            rewriter.create<mlir::scf::IfOp>(loc,
                new_result_types, adaptor.getCondition());
        rewriter.inlineRegionBefore(if_op.getThenRegion(),
                                    new_if_op.getThenRegion(),
                                    new_if_op.getThenRegion().end());
        if (mlir::failed(rewriter.convertRegionTypes(
                &new_if_op.getThenRegion(), *type_conv))) {
            return rewriter.notifyMatchFailure(if_op,
                "Converting then region types failed");
        }
        rewriter.inlineRegionBefore(if_op.getElseRegion(),
                                    new_if_op.getElseRegion(),
                                    new_if_op.getElseRegion().end());
        if (mlir::failed(rewriter.convertRegionTypes(
                &new_if_op.getElseRegion(), *type_conv))) {
            return rewriter.notifyMatchFailure(if_op,
                "Converting else region types failed");
        }
        rewriter.replaceOp(if_op, new_if_op.getResults());
        return mlir::success();
    }
};

// We have to manually convert the result type of scf.yield ops too
struct SCFYieldOpTypeFix : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
    using mlir::OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::scf::YieldOp yield,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yield, adaptor.getResults());
        return mlir::success();
    }
};

// We have to manually convert the result type of arith.select ops too
struct ArithSelectOpTypeFix : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
    using mlir::OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::arith::SelectOp select,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(select, adaptor.getCondition(),
                                                           adaptor.getTrueValue(),
                                                           adaptor.getFalseValue());
        return mlir::success();
    }
};

// We also have to manually convert the result type of a qcirc.calc
struct CalcOpTypeFix : public mlir::OpConversionPattern<qcirc::CalcOp> {
    using mlir::OpConversionPattern<qcirc::CalcOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qcirc::CalcOp calc,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(calc, "Need a type converter");
        }
        llvm::SmallVector<mlir::Type> new_result_types;
        if (mlir::failed(type_conv->convertTypes(
                calc->getResultTypes(), new_result_types))) {
            return rewriter.notifyMatchFailure(calc, "Could not convert result types");
        }

        mlir::Location loc = calc.getLoc();
        qcirc::CalcOp new_calc =
            rewriter.create<qcirc::CalcOp>(loc,
                new_result_types, adaptor.getInputs());
        rewriter.inlineRegionBefore(calc.getRegion(),
                                    new_calc.getRegion(),
                                    new_calc.getRegion().end());
        if (mlir::failed(rewriter.convertRegionTypes(
                &new_calc.getRegion(), *type_conv))) {
            return rewriter.notifyMatchFailure(calc,
                "Converting then region types failed");
        }
        rewriter.replaceOp(calc, new_calc.getResults());
        return mlir::success();
    }
};

// And finally we have to manually convert the result type of a
// qcirc.calc_yield as well
struct CalcYieldOpTypeFix : public mlir::OpConversionPattern<qcirc::CalcYieldOp> {
    using mlir::OpConversionPattern<qcirc::CalcYieldOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qcirc::CalcYieldOp yield,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::CalcYieldOp>(yield, adaptor.getResults());
        return mlir::success();
    }
};

struct QBundlePackOpLowering : public mlir::OpConversionPattern<qwerty::QBundlePackOp> {
    using mlir::OpConversionPattern<qwerty::QBundlePackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundlePackOp pack,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::ArrayPackOp>(pack, adaptor.getQubits());
        return mlir::success();
    }
};

struct QBundleUnpackOpLowering : public mlir::OpConversionPattern<qwerty::QBundleUnpackOp> {
    using mlir::OpConversionPattern<qwerty::QBundleUnpackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleUnpackOp unpack,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::ArrayUnpackOp>(unpack, adaptor.getQbundle());
        return mlir::success();
    }
};

struct BitBundlePackOpLowering : public mlir::OpConversionPattern<qwerty::BitBundlePackOp> {
    using mlir::OpConversionPattern<qwerty::BitBundlePackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::BitBundlePackOp pack,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::ArrayPackOp>(pack, adaptor.getBits());
        return mlir::success();
    }
};

struct BitBundleUnpackOpLowering : public mlir::OpConversionPattern<qwerty::BitBundleUnpackOp> {
    using mlir::OpConversionPattern<qwerty::BitBundleUnpackOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::BitBundleUnpackOp unpack,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::ArrayUnpackOp>(unpack, adaptor.getBundle());
        return mlir::success();
    }
};

// Since the function specialization work was already run at the beginning of
// this pass before we even started the dialect conversion engine, we can
// convert this qwerty.func directly to a func.func.
struct FuncOpLowering : public mlir::OpConversionPattern<qwerty::FuncOp> {
    using mlir::OpConversionPattern<qwerty::FuncOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncOp func,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(func, "Need a type converter");
        }
        mlir::Location loc = func.getLoc();
        mlir::FunctionType new_func_type = convertFuncType(func, *type_conv);
        mlir::func::FuncOp new_func =
            rewriter.create<mlir::func::FuncOp>(loc,
                func.getSymName(), new_func_type);
        if (func.isPrivate()) {
            new_func.setPrivate();
        }

        rewriter.inlineRegionBefore(func.getBody(),
                                    new_func.getBody(),
                                    new_func.getBody().end());

        if (mlir::failed(rewriter.convertRegionTypes(
                &new_func.getBody(), *type_conv))) {
            return rewriter.notifyMatchFailure(func,
                "Converting region types failed");
        }

        rewriter.eraseOp(func);
        return mlir::success();
    }
};

struct ReturnOpLowering : public mlir::OpConversionPattern<qwerty::ReturnOp> {
    using mlir::OpConversionPattern<qwerty::ReturnOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::ReturnOp ret,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(ret, adaptor.getOperands());
        return mlir::success();
    }
};

struct CallOpLowering : public mlir::OpConversionPattern<qwerty::CallOp> {
    using mlir::OpConversionPattern<qwerty::CallOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::CallOp call,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(call, "Need a type converter");
        }
        llvm::SmallVector<mlir::Type> new_result_types;
        if (mlir::failed(type_conv->convertTypes(
                call.getResults().getTypes(), new_result_types))) {
            return rewriter.notifyMatchFailure(call, "Could not convert result types");
        }

        if (call.getPred() && !call.getPredAttr().hasOnlyOnes()) {
            return rewriter.notifyMatchFailure(call,
                "Predicate is not all 1s. Is -only-pred-ones broken (or did "
                "it not run?)");
        }

        std::string new_sym_name = FuncSpec(call).getSymName(call.getCallee());
        rewriter.replaceOpWithNewOp<mlir::func::CallOp>(call,
            new_sym_name, new_result_types, adaptor.getCapturesAndOperands());
        return mlir::success();
    }
};

struct CallIndirectOpLowering : public mlir::OpConversionPattern<qwerty::CallIndirectOp> {
    using mlir::OpConversionPattern<qwerty::CallIndirectOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::CallIndirectOp call,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::CallableInvokeOp>(call,
            adaptor.getCallee(), adaptor.getCallOperands());
        return mlir::success();
    }
};

struct FuncConstOpLowering : public mlir::OpConversionPattern<qwerty::FuncConstOp> {
    using mlir::OpConversionPattern<qwerty::FuncConstOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncConstOp func_const,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        const mlir::TypeConverter *type_conv = getTypeConverter();
        if (!type_conv) {
            return rewriter.notifyMatchFailure(
                func_const, "Need a type converter");
        }
        qcirc::CallableType func_ty;
        if (!(func_ty = type_conv->convertType<qcirc::CallableType>(
                func_const.getResult().getType()))) {
            return rewriter.notifyMatchFailure(func_const,
                "Could not convert result type of qwerty.func_const to a "
                "CallableType");
        }
        std::string metadata_sym =
            FuncSpec::getMetadataSymName(func_const.getFunc());

        rewriter.replaceOpWithNewOp<qcirc::CallableCreateOp>(func_const,
            func_ty, metadata_sym, adaptor.getCaptures());
        return mlir::success();
    }
};

struct FuncAdjointOpLowering : public mlir::OpConversionPattern<qwerty::FuncAdjointOp> {
    using mlir::OpConversionPattern<qwerty::FuncAdjointOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncAdjointOp func_adj,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<qcirc::CallableAdjointOp>(func_adj,
            adaptor.getCallee());
        return mlir::success();
    }
};

struct FuncPredOpLowering : public mlir::OpConversionPattern<qwerty::FuncPredOp> {
    using mlir::OpConversionPattern<qwerty::FuncPredOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncPredOp func_pred,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        if (!func_pred.getPred().hasOnlyOnes()) {
            return rewriter.notifyMatchFailure(func_pred,
                "Predicate is not all 1s. Is -only-pred-ones broken (or did "
                "it not run?)");
        }

        rewriter.replaceOpWithNewOp<qcirc::CallableControlOp>(func_pred,
            adaptor.getCallee(), func_pred.getPred().getDim());
        return mlir::success();
    }
};

struct BitInitOpLowering : public mlir::OpConversionPattern<qwerty::BitInitOp> {
    using mlir::OpConversionPattern<qwerty::BitInitOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::BitInitOp prep,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = prep.getLoc();

        mlir::ValueRange bits = rewriter.create<qwerty::BitBundleUnpackOp>(loc, prep.getBitBundle()).getBits();
        mlir::ValueRange input_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, prep.getQbundleIn()).getQubits();
        llvm::SmallVector<mlir::Value> qubits(input_unpacked.begin(),
                                              input_unpacked.end());

        for (size_t i = 0; i < bits.size(); i++) {
            mlir::Value bit  = bits[i];
            mlir::Value qubit = qubits[i];

            mlir::scf::IfOp if_op = rewriter.create<mlir::scf::IfOp>(loc, rewriter.getType<qcirc::QubitType>(), bit, true);
            mlir::Block *then_block = if_op.thenBlock();
            mlir::Block *else_block = if_op.elseBlock();

            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(then_block);
                mlir::Value flipped = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubit).getResult();
                rewriter.create<mlir::scf::YieldOp>(loc, flipped);

                rewriter.setInsertionPointToStart(else_block);
                rewriter.create<mlir::scf::YieldOp>(loc, qubit);
            }

            mlir::ValueRange results = if_op.getResults();
            assert(results.size() == 1 && "too many if outputs");
            qubits[i] = results[0];
        }

        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(prep, qubits);
        return mlir::success();
    }
};

struct QBundleInitOpLowering : public mlir::OpConversionPattern<qwerty::QBundleInitOp> {
    using mlir::OpConversionPattern<qwerty::QBundleInitOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleInitOp init,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = init.getLoc();
        qwerty::BasisAttr basis = init.getBasis();
        mlir::ValueRange basis_phases = init.getBasisPhases();

        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, init.getQbundleIn()).getQubits();
        llvm::SmallVector<mlir::Value> qubits(unpacked.begin(), unpacked.end());
        if (basis.getDim() != qubits.size()) {
            return rewriter.notifyMatchFailure(init, "basis size does not match input bundle size");
        }

        uint64_t qubit_idx = 0;
        uint64_t phase_idx = 0;
        for (uint64_t i = 0; i < basis.getElems().size(); i++) {
            qwerty::BasisElemAttr elem = basis.getElems()[i];
            if (elem.getStd()) {
                return rewriter.notifyMatchFailure(init, "Passing a standard basis to qbinit makes no sense");
            } else if (elem.getVeclist()) {
                if (elem.getVeclist().getVectors().size() != 1) {
                    return rewriter.notifyMatchFailure(init, "Expected a singleton basis for qbinit");
                }
                qwerty::BasisVectorAttr vec = elem.getVeclist().getVectors()[0];
                if (vec.hasPhase()) {
                    qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                            loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]
                        ).getResult();
                    qubits[qubit_idx] = rewriter.create<qcirc::Gate1Q1POp>(
                            loc, qcirc::Gate1Q1P::P, basis_phases[phase_idx++], mlir::ValueRange(), qubits[qubit_idx]
                        ).getResult();
                    qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                            loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]
                        ).getResult();
                }
                for (uint64_t j = 0; j < vec.getDim(); j++) {
                    uint64_t bit = vec.getEigenbits()[vec.getDim()-j-1];
                    if (bit) {
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                    }
                    if (vec.getPrimBasis() == qwerty::PrimitiveBasis::X) {
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::H, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                    } else if (vec.getPrimBasis() == qwerty::PrimitiveBasis::Y) {
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::H, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::S, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                    }
                    qubit_idx++;
                }
            } else {
                return rewriter.notifyMatchFailure(init, "caught a mystery basis");
            }
        }

        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(init, qubits);
        return mlir::success();
    }
};

// This is basically QBundleInitOpLowering written backwards
struct QBundleDeinitOpLowering : public mlir::OpConversionPattern<qwerty::QBundleDeinitOp> {
    using mlir::OpConversionPattern<qwerty::QBundleDeinitOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleDeinitOp deinit,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = deinit.getLoc();
        qwerty::BasisAttr basis = deinit.getBasis();
        mlir::ValueRange basis_phases = deinit.getBasisPhases();

        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, deinit.getQbundleIn()).getQubits();
        llvm::SmallVector<mlir::Value> qubits(unpacked.begin(), unpacked.end());
        if (basis.getDim() != qubits.size()) {
            return rewriter.notifyMatchFailure(deinit, "basis size does not match input bundle size");
        }

        uint64_t qubit_idx = 0;
        uint64_t phase_idx = 0;
        for (uint64_t i = 0; i < basis.getElems().size(); i++) {
            qwerty::BasisElemAttr elem = basis.getElems()[i];
            if (elem.getStd()) {
                return rewriter.notifyMatchFailure(deinit, "Passing a standard basis to qbinit makes no sense");
            } else if (elem.getVeclist()) {
                if (elem.getVeclist().getVectors().size() != 1) {
                    return rewriter.notifyMatchFailure(deinit, "Expected a singleton basis for qbinit");
                }
                qwerty::BasisVectorAttr vec = elem.getVeclist().getVectors()[0];
                for (uint64_t j = 0; j < vec.getDim(); j++) {
                    uint64_t bit = vec.getEigenbits()[vec.getDim()-j-1];
                    if (vec.getPrimBasis() == qwerty::PrimitiveBasis::X) {
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::H, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                    } else if (vec.getPrimBasis() == qwerty::PrimitiveBasis::Y) {
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::Sdg, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::H, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                    }
                    if (bit) {
                        qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]
                            ).getResult();
                    }
                    qubit_idx++;
                }
                if (vec.hasPhase()) {
                    qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                            loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]
                        ).getResult();
                    mlir::Value neg_theta = wrapStationaryFloatOps(
                        rewriter, loc, basis_phases[phase_idx++],
                        [&](mlir::ValueRange args) {
                            assert(args.size() == 1);
                            mlir::Value theta_arg = args[0];
                            return rewriter.create<mlir::arith::NegFOp>(loc, theta_arg).getResult();
                        });
                    qubits[qubit_idx] = rewriter.create<qcirc::Gate1Q1POp>(
                            loc, qcirc::Gate1Q1P::P, neg_theta, mlir::ValueRange(), qubits[qubit_idx]
                        ).getResult();
                    qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(
                            loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]
                        ).getResult();
                }
            } else {
                return rewriter.notifyMatchFailure(deinit, "caught a mystery basis");
            }
        }

        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(deinit, qubits);
        return mlir::success();
    }
};

// Pattern for only '0'[N]
struct TrivialQBundlePrepOpLowering : public mlir::OpConversionPattern<qwerty::QBundlePrepOp> {
    using mlir::OpConversionPattern<qwerty::QBundlePrepOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundlePrepOp prep,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = prep.getLoc();
        if (prep.getPrimBasis() == qwerty::PrimitiveBasis::Z
                && prep.getEigenstate() == qwerty::Eigenstate::PLUS) {
            llvm::SmallVector<mlir::Value> qubits;
            qubits.reserve(prep.getDim());

            for (uint64_t i = 0; i < prep.getDim(); i++) {
                qubits.push_back(rewriter.create<qcirc::QallocOp>(loc).getResult());
            }

            // Hopefully canonicalization will get rid of this guy
            rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(prep, qubits);
            return mlir::success();
        } else {
            // NontrivialQBundlePrepOpLowering should take care of this
            return mlir::failure();
        }
    }
};

// For all qbprep ops besides the trivial '0'[N] case, produce a qbprep '0'[N]
// and then pass it through a qbinit op. This avoids some code duplication
// between here and qbinit lowering
struct NontrivialQBundlePrepOpLowering : public mlir::OpConversionPattern<qwerty::QBundlePrepOp> {
    using mlir::OpConversionPattern<qwerty::QBundlePrepOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundlePrepOp prep,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = prep.getLoc();
        // TrivialQBundlePrepOpLowering should take care of this
        if (prep.getPrimBasis() == qwerty::PrimitiveBasis::Z
                && prep.getEigenstate() == qwerty::Eigenstate::PLUS) {
            return mlir::failure();
        } else {
            mlir::Value zeros = rewriter.create<qwerty::QBundlePrepOp>(
                    loc, qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::PLUS, prep.getDim()
                ).getResult();
            qwerty::BasisAttr basis = rewriter.getAttr<qwerty::BasisAttr>(
                rewriter.getAttr<qwerty::BasisElemAttr>(
                    rewriter.getAttr<qwerty::BasisVectorListAttr>(
                        rewriter.getAttr<qwerty::BasisVectorAttr>(prep.getPrimBasis(), prep.getEigenstate(), prep.getDim(), false))));
            rewriter.replaceOpWithNewOp<qwerty::QBundleInitOp>(prep, basis, mlir::ValueRange(), zeros);
            return mlir::success();
        }
    }
};

struct QBundleDiscardOpLowering : public mlir::OpConversionPattern<qwerty::QBundleDiscardOp> {
    using mlir::OpConversionPattern<qwerty::QBundleDiscardOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleDiscardOp discard,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = discard.getLoc();
        uint64_t dim = discard.getQbundle().getType().getDim();
        // Hopefully canonicalization will get rid of this guy
        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, discard.getQbundle()).getQubits();
        for (uint64_t i = 0; i < dim; i++) {
            rewriter.create<qcirc::QfreeOp>(loc, unpacked[i]);
        }
        rewriter.eraseOp(discard);
        return mlir::success();
    }
};

struct QBundleDiscardZeroOpLowering : public mlir::OpConversionPattern<qwerty::QBundleDiscardZeroOp> {
    using mlir::OpConversionPattern<qwerty::QBundleDiscardZeroOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleDiscardZeroOp discard,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = discard.getLoc();
        uint64_t dim = discard.getQbundle().getType().getDim();
        // Hopefully canonicalization will get rid of this guy
        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, discard.getQbundle()).getQubits();
        for (uint64_t i = 0; i < dim; i++) {
            rewriter.create<qcirc::QfreeZeroOp>(loc, unpacked[i]);
        }
        rewriter.eraseOp(discard);
        return mlir::success();
    }
};

struct QBundlePhaseOpLowering : public mlir::OpConversionPattern<qwerty::QBundlePhaseOp> {
    using mlir::OpConversionPattern<qwerty::QBundlePhaseOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundlePhaseOp phaseOp,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = phaseOp.getLoc();
        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, phaseOp.getQbundleIn()).getQubits();
        mlir::Value victim = unpacked[0];
        victim = rewriter.create<qcirc::Gate1Q1POp>(loc, qcirc::Gate1Q1P::P, phaseOp.getTheta(), mlir::ValueRange(), victim).getResult();
        victim = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, mlir::ValueRange(), victim).getResult();
        victim = rewriter.create<qcirc::Gate1Q1POp>(loc, qcirc::Gate1Q1P::P, phaseOp.getTheta(), mlir::ValueRange(), victim).getResult();
        victim = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, mlir::ValueRange(), victim).getResult();
        llvm::SmallVector<mlir::Value> new_qubits;
        new_qubits.push_back(victim);
        new_qubits.append(unpacked.begin()+1, unpacked.end());
        // Hopefully canonicalization will get rid of this guy
        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(phaseOp, new_qubits);
        return mlir::success();
    }
};

struct QBundleIdentityOpLowering : public mlir::OpConversionPattern<qwerty::QBundleIdentityOp> {
    using mlir::OpConversionPattern<qwerty::QBundleIdentityOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleIdentityOp id,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOp(id, adaptor.getQbundleIn());
        return mlir::success();
    }
};

// See Section 2B of docs/state-prep.md
void computeAngles(std::complex<double> alpha,
                   std::complex<double> beta,
                   double &gamma,
                   double &theta,
                   double &phi) {
    double arg_alpha = std::arg(alpha);
    double arg_beta = std::arg(beta);

    gamma = (arg_alpha + arg_beta) / 2.0;
    if (std::abs(alpha) > ATOL) {
        phi = 2*(gamma - arg_alpha);
    } else { // std::abs(beta) > ATOL
        phi = 2*(arg_beta - gamma);
    }
    theta = 2.0 * std::acos(std::abs(alpha));
}

// See Section 3D of docs/state-prep.md
void revisedUndoSubroutine(
        std::vector<Eigen::SparseVector<double>> &thetas,
        std::vector<Eigen::SparseVector<double>> &phis,
        Eigen::SparseVector<std::complex<double>> &state_vector,
        uint64_t dim) {
    assert(dim && "dimension is not positive");

    if (dim == 1) {
        double theta = 0.0;
        double phi = 0.0;
        double gamma = 0.0;
        computeAngles(state_vector.coeff(0), state_vector.coeff(1), gamma, theta, phi);
        Eigen::SparseVector<double> theta_vec_1(1);
        Eigen::SparseVector<double> phi_vec_1(1);
        if (std::abs(theta - 0) > ATOL) {
            theta_vec_1.coeffRef(0) = theta;
            phi_vec_1.coeffRef(0) = phi;
        }
        thetas[dim - 1] = theta_vec_1;
        phis[dim - 1] = phi_vec_1;
    } else { // dim > 1
        Eigen::SparseVector<std::complex<double>> state_vector_prime(std::pow(2, dim - 1));
        Eigen::SparseVector<double> theta_vec_n(std::pow(2, dim - 1));
        Eigen::SparseVector<double> phi_vec_n(std::pow(2, dim - 1));
        size_t skip_idx = state_vector.size(); // Initialized with Size if no skipping was needed, no false skipping happens
        for (Eigen::SparseVector<std::complex<double>>::InnerIterator it(state_vector); it; ++it) {
            size_t idx = it.index();
            if (idx == skip_idx) {
                continue;
            }
            size_t j = idx >> 1;
            std::complex<double> alpha = it.value();
            std::complex<double> alpha_2j, alpha_2jplus1;
            if (idx & 1) { // odd case. alpha is alpha2jplus1 right now and alpha2j == 0
                alpha_2j = 0;
                alpha_2jplus1 = alpha;
            } else { // even case
                alpha_2j = alpha;
                alpha_2jplus1 = state_vector.coeff(idx+1);
                // Skip the next guy. we've already handled it!
                skip_idx = idx+1;
            }

            std::complex<double> alpha_2j_sqr = std::pow(std::abs(alpha_2j), 2);
            std::complex<double> alpha_2jplus1_sqr = std::pow(std::abs(alpha_2jplus1), 2);
            std::complex<double> r_j = std::sqrt(alpha_2j_sqr + alpha_2jplus1_sqr);

            std::complex<double> alpha_2j_prime = alpha_2j / r_j;
            std::complex<double> alpha_2jplus1_prime = alpha_2jplus1 / r_j;
            double gamma_j = 0.0;
            double theta_j = 0.0;
            double phi_j = 0.0;
            computeAngles(alpha_2j_prime, alpha_2jplus1_prime, gamma_j, theta_j, phi_j);
            state_vector_prime.coeffRef(j) = r_j * std::polar(1.0, gamma_j);

            // Add gates if necessary to undo
            if (std::abs(theta_j - 0) <= ATOL) continue;
            theta_vec_n.coeffRef(j) = theta_j;
            phi_vec_n.coeffRef(j) = phi_j;
        }
        thetas[dim - 1] = theta_vec_n;
        phis[dim - 1] = phi_vec_n;
        revisedUndoSubroutine(thetas, phis, state_vector_prime, dim - 1);
    }
}

// See Section 3B of docs/state-prep.md
void recrunchAngles(
        Eigen::SparseVector<double> &angles_in,
        Eigen::SparseVector<double> &angles_out,
        std::function<double(double, double)> angle_computer) {
    size_t skip_idx = angles_in.size();
    for (Eigen::SparseVector<double>::InnerIterator it(angles_in); it; ++it) {
        size_t idx = it.index();
        if (idx == skip_idx) {
            continue;
        }
        size_t k = idx >> 1;
        double angle = it.value();
        double angle_2k, angle_2kplus1;
        if (idx & 1) { // odd case
            angle_2k = 0;
            angle_2kplus1 = angle;
        } else { // even case
            angle_2k = angle;
            angle_2kplus1 = angles_in.coeff(idx+1);
            // Skip the next guy. we've already handled it!
            skip_idx = idx+1;
        }

        double angle_k_prime = angle_computer(angle_2k, angle_2kplus1);
        angles_out.coeffRef(k) = angle_k_prime;
    }
}

// Checks if angles has an angle that is not very close to 0
bool hasNontrivialAngle(Eigen::SparseVector<double> &angles) {
    if (!angles.nonZeros()) {
        return false;
    }

    // Why didn't angles.nonZeros() cover it above? Because there could be a
    // bunch of angles that are e.g. 1e-10
    bool found_nonzero = false;
    for (Eigen::SparseVector<double>::InnerIterator it(angles); it; ++it) {
        if (std::abs(it.value()) > ATOL) {
            found_nonzero = true;
            break;
        }
    }

    return found_nonzero;
}

// Due to Shende et al.: Theorem 8 of https://doi.org/10.1109/TCAD.2005.855930
void multiplexedRyRzSynthesis(
        mlir::RewriterBase &rewriter,
        mlir::Location &loc,
        qcirc::Gate1Q1P rot_kind,
        uint64_t dim,
        llvm::SmallVector<mlir::Value> &qubits,
        Eigen::SparseVector<double> &thetas) {
    assert(dim && "qubit dimension is not positive");

    if (!hasNontrivialAngle(thetas)) {
        // Nothing to do here, all angles are 0s
        return;
    }

    llvm::SmallVector<mlir::Value> no_controls;
    if (dim == 1) {
        mlir::Value theta_const =
            qcirc::stationaryF64Const(rewriter, loc, thetas.coeff(0));
        qubits[0] = rewriter.create<qcirc::Gate1Q1POp>(loc, rot_kind, theta_const, no_controls, qubits[0]).getResult();
    } else { // dim > 1
        {
            Eigen::SparseVector<double> thetas_prime(thetas.size() / 2);
            recrunchAngles(thetas, thetas_prime,
                           [](double theta_2k, double theta_2kplus1) { return (theta_2k + theta_2kplus1) / 2; });
            // Exclude second-to-last qubit
            llvm::SmallVector<mlir::Value> qubits_prime(qubits.begin(), qubits.begin() + qubits.size()-2);
            qubits_prime.push_back(qubits[qubits.size()-1]);
            multiplexedRyRzSynthesis(rewriter, loc, rot_kind, dim-1, qubits_prime, thetas_prime);
            // Thread qubits through
            for (size_t i = 0; i < qubits_prime.size()-1; i++) {
                qubits[i] = qubits_prime[i];
            }
            qubits[qubits.size()-1] = qubits_prime[qubits_prime.size()-1];
        }

        qcirc::Gate1QOp X = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, qubits[dim - 2], qubits[dim - 1]);
        qubits[dim - 2] = X.getControlResults()[0];
        qubits[dim - 1] = X.getResult();

        {
            Eigen::SparseVector<double> thetas_double_prime(thetas.size() / 2);
            recrunchAngles(thetas, thetas_double_prime,
                           [](double theta_2k, double theta_2kplus1) { return (theta_2k - theta_2kplus1) / 2; });
            // Exclude second-to-last qubit
            llvm::SmallVector<mlir::Value> qubits_double_prime(qubits.begin(), qubits.begin() + qubits.size()-2);
            qubits_double_prime.push_back(qubits[qubits.size()-1]);
            multiplexedRyRzSynthesis(rewriter, loc, rot_kind, dim-1, qubits_double_prime, thetas_double_prime);
            // Thread qubits through
            for (size_t i = 0; i < qubits_double_prime.size()-1; i++) {
                qubits[i] = qubits_double_prime[i];
            }
            qubits[qubits.size()-1] = qubits_double_prime[qubits_double_prime.size()-1];
        }

        X = rewriter.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, qubits[dim - 2], qubits[dim - 1]);
        qubits[dim - 2] = X.getControlResults()[0];
        qubits[dim - 1] = X.getResult();
    }
}

// Due to Shende et al.: Theorem 9 of https://doi.org/10.1109/TCAD.2005.855930
void arbitraryStatePrep(mlir::RewriterBase &rewriter,
                        mlir::Location loc,
                        qwerty::SuperposAttr superpos,
                        llvm::SmallVectorImpl<mlir::Value> &qubits) {
    size_t dim = superpos.getDim();
    Eigen::SparseVector<std::complex<double>> state_vector(1ULL << dim);

    for (qwerty::SuperposElemAttr elem : superpos.getElems()) {
        double prob = elem.getProb().getValueAsDouble();
        double phase = elem.getPhase().getValueAsDouble();
        llvm::APInt eigenbits = elem.getEigenbits();
        state_vector.coeffRef(eigenbits.getZExtValue()) = std::sqrt(prob) * std::polar(1.0, phase);
    }

    // Run Undo Subroutine to get list of theta and phi vectors
    std::vector<Eigen::SparseVector<double>> theta_vecs(dim);
    std::vector<Eigen::SparseVector<double>> phi_vecs(dim);
    revisedUndoSubroutine(theta_vecs, phi_vecs, state_vector, dim);

    for (size_t i = 1; i <= dim; i++) {
        llvm::SmallVector<mlir::Value> target_qubits(qubits.begin(), qubits.begin()+i);

        multiplexedRyRzSynthesis(rewriter, loc, qcirc::Gate1Q1P::Ry, i, target_qubits, theta_vecs[i - 1]);
        multiplexedRyRzSynthesis(rewriter, loc, qcirc::Gate1Q1P::Rz, i, target_qubits, phi_vecs[i - 1]);

        for (size_t i = 0; i < target_qubits.size(); i++) {
            qubits[i] = target_qubits[i];
        }
    }
}

// Check if the superposition desired is an equal superposition of all
// computational basis states. If all basis states have the same phase, then it
// is ignored (since it is a global phase)
bool isUniformSuperpos(qwerty::SuperposAttr superpos) {
    size_t dim = superpos.getDim();

    if (superpos.getElems().size() != (1ULL << dim)) {
        return false;
    }

    double uniform_prob = 1.0/(1ULL << dim);
    bool first = true;
    double prev_phase;

    for (qwerty::SuperposElemAttr elem : superpos.getElems()) {
        double phase = elem.getPhase().getValueAsDouble();
        if (first) {
            prev_phase = phase;
        } else if (std::abs(prev_phase - phase) > ATOL) {
            return false;
        }

        double prob = elem.getProb().getValueAsDouble();
        if (std::abs(prob - uniform_prob) > ATOL) {
            return false;
        }

        first = false;
    }

    // The phase is a global phase. We are good to prepare this as an equal
    // superposition
    return true;
}

// Prepare a uniform state with a broadcast of Hadamards. Qiskit does a similar
// optimization to this
void uniformStatePrep(mlir::RewriterBase &rewriter,
                      mlir::Location loc,
                      llvm::SmallVectorImpl<mlir::Value> &qubits) {
    for (size_t i = 0; i < qubits.size(); i++) {
        qubits[i] = rewriter.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::H, mlir::ValueRange(), qubits[i]).getResult();
    }
}

// Check whether the desired superposition is of the form
//     cos(θ/2)e^{iΦ_1}|x⟩ + sin(θ/2)e^{iΦ_2}|¬x⟩
// where ¬ is the bitwise complement. We call this a "quasi-GHZ state." It is
// cheap to prepare (see below)
bool isQuasiGhzState(qwerty::SuperposAttr superpos) {
    if (superpos.getElems().size() != 2) {
        return false;
    }

    llvm::APInt differing_eigenbits = superpos.getElems()[0].getEigenbits();
    differing_eigenbits ^= superpos.getElems()[1].getEigenbits();
    return differing_eigenbits.isAllOnes();
}

// Prepare a quasi-GHZ state (defined above for isQuasiGhzState())
// with the gates Ry(θ), a ladder of CNOTs, some X gates, and then an Rz gate.
void quasiGhzStatePrep(mlir::RewriterBase &rewriter,
                       mlir::Location loc,
                       qwerty::SuperposAttr superpos,
                       llvm::SmallVectorImpl<mlir::Value> &qubits) {
    assert(superpos.getElems().size() == 2
           && "expected 2 superpos elems for ghz");
    size_t dim = qubits.size();
    assert(dim == superpos.getDim() && "wrong size of qubit array");

    qwerty::SuperposElemAttr elem0 = superpos.getElems()[0];
    qwerty::SuperposElemAttr elem1 = superpos.getElems()[1];

    // elem0's first eigenbit is 0 and elem1's first eigenbit is 1
    if (elem0.getVectors()[0].getEigenbits().isSignBitSet()) { // if msb==1
        std::swap(elem0, elem1);
    }

    // First, get the right absolute value of amplitudes
    double theta = 2*std::acos(std::sqrt(elem0.getProb().getValueAsDouble()));
    mlir::Value theta_const =
        qcirc::stationaryF64Const(rewriter, loc, theta);
    qubits[0] = rewriter.create<qcirc::Gate1Q1POp>(
            loc, qcirc::Gate1Q1P::Ry, theta_const, mlir::ValueRange(), qubits[0]
        ).getResult();

    // Now entangle all the qubits
    assert(dim && "number of superpos qubits must be positive");
    for (size_t i = 0; i < dim-1; i++) {
        qcirc::Gate1QOp cnot = rewriter.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::X, qubits[i], qubits[i+1]);

        assert(cnot.getControlResults().size() == 1);
        qubits[i] = cnot.getControlResults()[0];
        qubits[i+1] = cnot.getResult();
    }

    // Next flip bits as needed, giving us the desired state up to a phase
    llvm::APInt eigenbits0 = elem0.getEigenbits();
    for (size_t i = 1; i < dim; i++) {
        if (eigenbits0[eigenbits0.getBitWidth()-1 - i]) {
            qubits[i] = rewriter.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[i]).getResult();
        }
    }

    // Finally, impart phases if needed
    double phase0 = elem0.getPhase().getValueAsDouble();
    double phase1 = elem1.getPhase().getValueAsDouble();

    // Tiny optimization: notice that
    // e^{i\phi_0}|00⟩ + e^{i\phi_1}|00⟩
    // = e^{-i\phi_0}(|00⟩ + e^{i(\phi_1 - phi_0)}|11⟩).
    // Thus, we can apply a phase to just one of the states, not both
    double relative_phase = phase1 - phase0;

    if (std::abs(relative_phase - 0.0) >= ATOL) {
        mlir::Value phi_const =
            qcirc::stationaryF64Const(rewriter, loc, relative_phase);
        qubits[0] = rewriter.create<qcirc::Gate1Q1POp>(
                loc, qcirc::Gate1Q1P::P, phi_const, mlir::ValueRange(), qubits[0]
            ).getResult();
    }
}

struct SuperposOpLowering : public mlir::OpConversionPattern<qwerty::SuperposOp> {
    using mlir::OpConversionPattern<qwerty::SuperposOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::SuperposOp superPosOp,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = superPosOp.getLoc();
        size_t dim = adaptor.getSuperpos().getDim();
        llvm::SmallVector<mlir::Value> qubits;
        qubits.reserve(dim);

        // Allocate zero qubits
        for (uint64_t i = 0; i < dim; i++) {
            qubits.push_back(rewriter.create<qcirc::QallocOp>(loc).getResult());
        }

        qwerty::SuperposAttr superpos = superPosOp.getSuperpos();
        if (isUniformSuperpos(superpos)) {
            uniformStatePrep(rewriter, loc, qubits);
        } else if (isQuasiGhzState(superpos)) {
            quasiGhzStatePrep(rewriter, loc, superpos, qubits);
        } else {
            // Fall back to general case
            arbitraryStatePrep(rewriter, loc, superpos, qubits);
        }

        mlir::Value qbundle = rewriter.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();

        qwerty::BasisAttr std_N = rewriter.getAttr<qwerty::BasisAttr>(
            std::initializer_list<qwerty::BasisElemAttr>{
            rewriter.getAttr<qwerty::BasisElemAttr>(
                rewriter.getAttr<qwerty::BuiltinBasisAttr>(qwerty::PrimitiveBasis::Z, dim))});
        llvm::SmallVector<qwerty::BasisElemAttr> desired_prim_bases;
        auto vecs = adaptor.getSuperpos().getElems()[0].getVectors();
        for (qwerty::BasisVectorAttr elem_vec : vecs) {
            desired_prim_bases.push_back(rewriter.getAttr<qwerty::BasisElemAttr>(
                rewriter.getAttr<qwerty::BuiltinBasisAttr>(elem_vec.getPrimBasis(), elem_vec.getDim())));
        }
        qwerty::BasisAttr desired_basis = rewriter.getAttr<qwerty::BasisAttr>(desired_prim_bases);
        rewriter.replaceOpWithNewOp<qwerty::QBundleBasisTranslationOp>(
            superPosOp, std_N, desired_basis, mlir::ValueRange(), qbundle);
        return mlir::success();
    }
};

struct EnsembleOpLowering : public mlir::OpConversionPattern<qwerty::EnsembleOp> {
    using mlir::OpConversionPattern<qwerty::EnsembleOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::EnsembleOp ensemble,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = ensemble.getLoc();
        qwerty::SuperposAttr superpos_attr = ensemble.getSuperpos();
        size_t n_superpos_elems = superpos_attr.getElems().size();
        size_t n_extra_bits = BITS_NEEDED(n_superpos_elems);

        llvm::SmallVector<qwerty::SuperposElemAttr> new_superpos_elems;
        for (auto [i, superpos_elem] : llvm::enumerate(superpos_attr.getElems())) {
            llvm::SmallVector<qwerty::BasisVectorAttr> new_vectors;
            llvm::APInt extra_bits(n_extra_bits, i);
            new_vectors.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
                qwerty::PrimitiveBasis::Z, extra_bits, n_extra_bits, false));
            auto old_vectors = superpos_elem.getVectors();
            new_vectors.append(old_vectors.begin(), old_vectors.end());
            qwerty::SuperposElemAttr new_superpos_elem =
                rewriter.getAttr<qwerty::SuperposElemAttr>(
                    superpos_elem.getProb(),
                    superpos_elem.getPhase(),
                    new_vectors
                );
            new_superpos_elems.push_back(new_superpos_elem);
        }

        qwerty::SuperposAttr new_superpos_attr =
            rewriter.getAttr<qwerty::SuperposAttr>(new_superpos_elems);
        mlir::Value superpos = rewriter.create<qwerty::SuperposOp>(
            loc, new_superpos_attr).getQbundle();

        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(
            loc, superpos).getQubits();
        assert(unpacked.size() > n_extra_bits
               && "Superpos bundle is too small");

        llvm::SmallVector<mlir::Value> extra_qubits(
            unpacked.begin(),
            unpacked.begin() + n_extra_bits);
        llvm::SmallVector<mlir::Value> ensemble_qubits(
            unpacked.begin() + n_extra_bits,
            unpacked.end());

        mlir::Value extra_qbundle = rewriter.create<qwerty::QBundlePackOp>(
            loc, extra_qubits).getQbundle();
        mlir::Value ensemble_qbundle = rewriter.create<qwerty::QBundlePackOp>(
            loc, ensemble_qubits).getQbundle();

        rewriter.create<qwerty::QBundleDiscardOp>(loc, extra_qbundle);
        rewriter.replaceOp(ensemble, ensemble_qbundle);
        return mlir::success();
    }
};

// As specified in Section 5.1 of Mike & Ike
void runQft(mlir::Location loc, mlir::OpBuilder &builder,
            llvm::SmallVectorImpl<mlir::Value> &control_qubits,
            llvm::SmallVectorImpl<mlir::Value> &qubits,
            size_t start_idx, size_t n) {
    assert(n <= qubits.size() && start_idx < qubits.size()
           && start_idx+n <= qubits.size()
           && "QFT indices out of range");
    for (size_t i = 0; i < n; i++) {
        qcirc::Gate1QOp h = builder.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::H, control_qubits,
            qubits[start_idx+i]);
        qubits[start_idx+i] = h.getResult();
        control_qubits.clear();
        control_qubits.append(h.getControlResults().begin(),
                              h.getControlResults().end());

        for (size_t j = i+1; j < n; j++) {
            // TODO: this precision is not enough for very large circuits. Need
            // to use llvm::APFloat or something
            double phase = 2.0*M_PI/std::pow(2, j-i+1);
            mlir::Value phase_const =
                qcirc::stationaryF64Const(builder, loc, phase);
            llvm::SmallVector<mlir::Value> p_controls(control_qubits.begin(),
                                                      control_qubits.end());
            p_controls.push_back(qubits[start_idx+j]);
            qcirc::Gate1Q1POp cp = builder.create<qcirc::Gate1Q1POp>(
                    loc, qcirc::Gate1Q1P::P, phase_const,
                    p_controls, qubits[start_idx+i]);
            control_qubits.clear();
            // Exclude last control qubit
            control_qubits.append(cp.getControlResults().begin(),
                                  cp.getControlResults().begin()
                                  + (cp.getControlResults().size()-1));
            // ...since we need it here
            qubits[start_idx+j] = cp.getControlResults()[
                cp.getControlResults().size()-1];
            qubits[start_idx+i] = cp.getResult();
        }
    }

    for (size_t i = 0; i < n/2; i++) {
        qcirc::Gate2QOp swap = builder.create<qcirc::Gate2QOp>(
                loc, qcirc::Gate2Q::Swap, control_qubits,
                qubits[start_idx+i], qubits[start_idx+n-1-i]);
        qubits[start_idx+i] = swap.getLeftResult();
        qubits[start_idx+n-1-i] = swap.getRightResult();
        control_qubits.clear();
        control_qubits.append(swap.getControlResults().begin(),
                              swap.getControlResults().end());
    }
}

// The above but written backwards (and the phases flipped), basically
void runInverseQft(mlir::Location loc, mlir::OpBuilder &builder,
                   llvm::SmallVectorImpl<mlir::Value> &control_qubits,
                   llvm::SmallVectorImpl<mlir::Value> &qubits,
                   size_t start_idx, size_t n) {
    assert(n <= qubits.size() && start_idx < qubits.size()
           && start_idx+n <= qubits.size()
           && "IQFT indices out of range");

    for (size_t ii = 0; ii < n/2; ii++) {
        size_t i = n/2-1-ii;
        qcirc::Gate2QOp swap = builder.create<qcirc::Gate2QOp>(
                loc, qcirc::Gate2Q::Swap, control_qubits,
                qubits[start_idx+i], qubits[start_idx+n-1-i]);
        qubits[start_idx+i] = swap.getLeftResult();
        qubits[start_idx+n-1-i] = swap.getRightResult();
        control_qubits.clear();
        control_qubits.append(swap.getControlResults().begin(),
                              swap.getControlResults().end());
    }

    for (size_t ii = 0; ii < n; ii++) {
        size_t i = n-1-ii;
        for (size_t jj = i+1; jj < n; jj++) {
            size_t j = n+i-jj;
            // TODO: this precision is not enough for very large circuits. Need
            // to use llvm::APFloat or something
            double phase = -2.0*M_PI/std::pow(2, j-i+1);
            mlir::Value phase_const =
                qcirc::stationaryF64Const(builder, loc, phase);
            llvm::SmallVector<mlir::Value> p_controls(control_qubits.begin(),
                                                      control_qubits.end());
            p_controls.push_back(qubits[start_idx+j]);
            qcirc::Gate1Q1POp cp = builder.create<qcirc::Gate1Q1POp>(
                    loc, qcirc::Gate1Q1P::P, phase_const,
                    p_controls, qubits[start_idx+i]);
            // Exclude last control qubit
            control_qubits.clear();
            control_qubits.append(cp.getControlResults().begin(),
                                  cp.getControlResults().begin()
                                  + (cp.getControlResults().size()-1));
            // ...since we need it here
            qubits[start_idx+j] = cp.getControlResults()[
                cp.getControlResults().size()-1];
            qubits[start_idx+i] = cp.getResult();
        }
        qcirc::Gate1QOp h = builder.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::H, control_qubits, qubits[start_idx+i]);
        qubits[start_idx+i] = h.getResult();
        control_qubits.clear();
        control_qubits.append(h.getControlResults().begin(),
                              h.getControlResults().end());
    }
}

// In the code here, we happen to use a stricter definition of "aligned" that
// means "aligned" (from the CGO paper) and the primitive basis is std.
bool isAligned(qwerty::BasisAttr lhs, qwerty::BasisAttr rhs) {
    if (lhs.getElems().size() != rhs.getElems().size()) {
        return false;
    }

    size_t n_elems = lhs.getElems().size();

    for (size_t i = 0; i < n_elems; i++) {
        qwerty::BasisElemAttr left = lhs.getElems()[i];
        qwerty::BasisElemAttr right = rhs.getElems()[i];

        if (left.getDim() != right.getDim()) {
            return false;
        }

        qwerty::BuiltinBasisAttr lstd = left.getStd();
        qwerty::BuiltinBasisAttr rstd = right.getStd();
        qwerty::BasisVectorListAttr lvl = left.getVeclist();
        qwerty::BasisVectorListAttr rvl = right.getVeclist();

        if (lstd && rstd) {
            if (lstd.getPrimBasis() != qwerty::PrimitiveBasis::Z
                    || rstd.getPrimBasis() != qwerty::PrimitiveBasis::Z) {
                return false;
            }
        } else if (lvl && rvl) {
            if (lvl.hasPhases() || rvl.hasPhases()
                    || lvl.getPrimBasis() != qwerty::PrimitiveBasis::Z
                    || rvl.getPrimBasis() != qwerty::PrimitiveBasis::Z) {
                return false;
            }
        } else {
            return false;
        }
    }

    return true;
}

// Extra validation to make sure span checking did not fail
bool isPermutation(llvm::ArrayRef<qwerty::BasisVectorAttr> left,
                   llvm::ArrayRef<qwerty::BasisVectorAttr> right) {
    if (left.size() != right.size()) {
        return false;
    }

    assert(!left.empty() && !right.empty() && "Empty basis literal?");

    if (left[0].getDim() != right[0].getDim()) {
        return false;
    }

    llvm::DenseSet<llvm::APInt> vecs;
    for (qwerty::BasisVectorAttr vec : left) {
        vecs.insert(vec.getEigenbits());
    }
    for (qwerty::BasisVectorAttr vec : right) {
        if (!vecs.count(vec.getEigenbits())) {
            return false;
        }
    }
    return true;
}

qwerty::BasisElemAttr rebuildZ(mlir::RewriterBase &rewriter,
                               qwerty::BuiltinBasisAttr bib) {
    if (bib.getPrimBasis() == qwerty::PrimitiveBasis::Z) {
        return rewriter.getAttr<qwerty::BasisElemAttr>(bib);
    } else {
        return rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BuiltinBasisAttr>(
                qwerty::PrimitiveBasis::Z, bib.getDim()));
    }
}

qwerty::BasisElemAttr rebuildZ(mlir::RewriterBase &rewriter,
                               qwerty::BasisVectorListAttr veclist) {
    if (veclist.getPrimBasis() == qwerty::PrimitiveBasis::Z && !veclist.hasPhases()) {
        return rewriter.getAttr<qwerty::BasisElemAttr>(veclist);
    } else {
        llvm::SmallVector<qwerty::BasisVectorAttr> vecs;
        for (qwerty::BasisVectorAttr vec : veclist.getVectors()) {
            vecs.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
                qwerty::PrimitiveBasis::Z, vec.getEigenbits(), vec.getDim(),
                /*hasPhase=*/false));
        }

        return rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BasisVectorListAttr>(vecs));
    }
}

qwerty::BuiltinBasisAttr splitStd(mlir::RewriterBase &rewriter,
                                  std::deque<qwerty::BasisElemAttr> &dest_queue,
                                  qwerty::BuiltinBasisAttr bib,
                                  size_t new_dim) {
    assert(bib.getDim() > new_dim && "built-in basis too small to split");

    qwerty::BuiltinBasisAttr ret =
        rewriter.getAttr<qwerty::BuiltinBasisAttr>(bib.getPrimBasis(), new_dim);

    qwerty::BasisElemAttr remainder =
        rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BuiltinBasisAttr>(
                bib.getPrimBasis(), bib.getDim() - new_dim));
    dest_queue.push_front(remainder);

    return ret;
}

// See factorFullOutOfVeclist() in ast.cpp. This is a little more complicated,
// though, because unlike in span checking, ordering matters here.
qwerty::BasisElemAttr factorFull(mlir::RewriterBase &rewriter,
                                 std::deque<qwerty::BasisElemAttr> &dest_queue,
                                 qwerty::BasisVectorListAttr vl,
                                 size_t std_dim) {
    size_t n_qubits = vl.getDim();
    size_t two_to_the_n = 1ULL << std_dim;
    if (vl.getVectors().size() % two_to_the_n) {
        return nullptr;
    }
    size_t n_suffixes = vl.getVectors().size() >> std_dim;

    // Whether we should return built-in basis or a veclist
    bool prefixes_in_order = true;
    llvm::SmallVector<llvm::APInt> prefix_order;
    prefix_order.reserve(two_to_the_n);

    llvm::SmallVector<bool> prefixes_seen(two_to_the_n, false);
    for (qwerty::BasisVectorAttr vec : vl.getVectors()) {
        llvm::APInt eigenbits = vec.getEigenbits();
        size_t prefix = eigenbits.extractBitsAsZExtValue(std_dim,
                                                         n_qubits - std_dim);
        llvm::APInt prefix_ap = eigenbits.extractBits(std_dim,
                                                      n_qubits - std_dim);
        if (!prefixes_seen[prefix]) {
            if (!prefix_order.empty()
                    && prefix_ap.ult(prefix_order[prefix_order.size()-1])) {
                prefixes_in_order = false;
            }
            prefix_order.push_back(eigenbits.extractBits(std_dim,
                                                         n_qubits - std_dim));
        }
        prefixes_seen[prefix] = true;
    }

    for (bool seen : prefixes_seen) {
        if (!seen) {
            return nullptr;
        }
    }

    llvm::SmallVector<llvm::APInt> suffix_order;
    suffix_order.reserve(n_suffixes);

    size_t i = 0;
    for (qwerty::BasisVectorAttr vec : vl.getVectors()) {
        llvm::APInt eigenbits = vec.getEigenbits();
        llvm::APInt prefix = eigenbits.extractBits(std_dim,
                                                   n_qubits - std_dim);
        llvm::APInt suffix = eigenbits.trunc(n_qubits - std_dim);

        if (prefix != prefix_order[i / n_suffixes]) {
            return nullptr;
        }

        if (i < n_suffixes) {
            suffix_order.push_back(suffix);
        } else if (suffix != suffix_order[i % n_suffixes]) {
            return nullptr;
        }

        i++;
    }

    qwerty::BasisElemAttr ret;
    if (prefixes_in_order) {
        ret = rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BuiltinBasisAttr>(vl.getPrimBasis(),
                                                        std_dim));
    } else {
        llvm::SmallVector<qwerty::BasisVectorAttr> vecs;
        vecs.reserve(two_to_the_n);
        for (llvm::APInt eigenbits : prefix_order) {
            vecs.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
                vl.getPrimBasis(), eigenbits, std_dim, /*hasPhase=*/false));
        }

        ret = rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BasisVectorListAttr>(vecs));
    }

    llvm::SmallVector<qwerty::BasisVectorAttr> suffix_vecs;
    suffix_vecs.reserve(n_suffixes);
    for (llvm::APInt eigenbits : suffix_order) {
        suffix_vecs.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
            vl.getPrimBasis(), eigenbits, n_qubits - std_dim, /*hasPhase=*/false));
    }
    qwerty::BasisElemAttr remainder =
        rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BasisVectorListAttr>(suffix_vecs));
    dest_queue.push_front(remainder);

    return ret;
}

qwerty::BasisVectorListAttr factorVeclist(
        mlir::RewriterBase &rewriter,
        std::deque<qwerty::BasisElemAttr> &dest_queue,
        qwerty::BasisVectorListAttr vl,
        qwerty::BasisVectorListAttr factor) {
    size_t n_qubits = vl.getDim();
    size_t n_factor_qubits = factor.getDim();
    size_t n_prefixes = factor.getVectors().size();
    if (vl.getVectors().size() % n_prefixes) {
        return nullptr;
    }
    size_t n_suffixes = vl.getVectors().size() / n_prefixes;

    llvm::DenseSet<llvm::APInt> allowed_prefixes;
    llvm::SmallVector<llvm::APInt> prefix_order;
    prefix_order.reserve(n_prefixes);
    for (qwerty::BasisVectorAttr vec : factor.getVectors()) {
        allowed_prefixes.insert(vec.getEigenbits());
        prefix_order.push_back(vec.getEigenbits());
    }

    llvm::DenseSet<llvm::APInt> prefixes_seen;
    for (qwerty::BasisVectorAttr vec : vl.getVectors()) {
        llvm::APInt eigenbits = vec.getEigenbits();
        llvm::APInt prefix = eigenbits.extractBits(n_factor_qubits,
                                                   n_qubits - n_factor_qubits);
        if (!allowed_prefixes.count(prefix)) {
            return nullptr;
        }
        prefixes_seen.insert(prefix);
    }

    for (llvm::APInt prefix : prefix_order) {
        if (!prefixes_seen.count(prefix)) {
            return nullptr;
        }
    }

    llvm::SmallVector<llvm::APInt> suffix_order;
    suffix_order.reserve(n_suffixes);

    size_t i = 0;
    for (qwerty::BasisVectorAttr vec : vl.getVectors()) {
        llvm::APInt eigenbits = vec.getEigenbits();
        llvm::APInt prefix = eigenbits.extractBits(n_factor_qubits,
                                                   n_qubits - n_factor_qubits);
        llvm::APInt suffix = eigenbits.trunc(n_qubits - n_factor_qubits);

        if (prefix != prefix_order[i / n_suffixes]) {
            return nullptr;
        }

        if (i < n_suffixes) {
            suffix_order.push_back(suffix);
        } else if (suffix != suffix_order[i % n_suffixes]) {
            return nullptr;
        }

        i++;
    }

    llvm::SmallVector<qwerty::BasisVectorAttr> prefix_vecs;
    prefix_vecs.reserve(n_prefixes);
    for (llvm::APInt eigenbits : prefix_order) {
        prefix_vecs.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
            vl.getPrimBasis(), eigenbits, n_factor_qubits, /*hasPhase=*/false));
    }
    qwerty::BasisVectorListAttr ret =
        rewriter.getAttr<qwerty::BasisVectorListAttr>(prefix_vecs);

    llvm::SmallVector<qwerty::BasisVectorAttr> suffix_vecs;
    suffix_vecs.reserve(n_suffixes);
    for (llvm::APInt eigenbits : suffix_order) {
        suffix_vecs.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
            vl.getPrimBasis(), eigenbits, n_qubits - n_factor_qubits,
            /*hasPhase=*/false));
    }
    qwerty::BasisElemAttr remainder =
        rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BasisVectorListAttr>(suffix_vecs));
    dest_queue.push_front(remainder);

    return ret;
}

qwerty::BasisVectorListAttr merge(
        mlir::RewriterBase &rewriter,
        qwerty::BasisVectorListAttr left,
        qwerty::BasisVectorListAttr right) {
    size_t new_dim = left.getDim() + right.getDim();
    llvm::SmallVector<qwerty::BasisVectorAttr> vecs;
    vecs.reserve(left.getVectors().size() * right.getVectors().size());

    for (qwerty::BasisVectorAttr lvec : left.getVectors()) {
        for (qwerty::BasisVectorAttr rvec : right.getVectors()) {
            llvm::APInt eigenbits = lvec.getEigenbits()
                                        .concat(rvec.getEigenbits());
            vecs.push_back(rewriter.getAttr<qwerty::BasisVectorAttr>(
                qwerty::PrimitiveBasis::Z, eigenbits, new_dim, false));
        }
    }

    return rewriter.getAttr<qwerty::BasisVectorListAttr>(vecs);
}

qwerty::BasisVectorListAttr merge(
        mlir::RewriterBase &rewriter,
        qwerty::BasisVectorListAttr left,
        qwerty::BasisElemAttr right) {
    qwerty::BasisVectorListAttr vl_right = right.getVeclist();
    if (!vl_right) {
        vl_right = right.getStd().expandToVeclist();
    }
    return merge(rewriter, left, vl_right);
}

// Mission: merge small with its successors until it's the same dimension as
// big. (In the process, they may swap roles)
bool greedyMerge(mlir::RewriterBase &rewriter,
                 qwerty::BasisVectorListAttr big,
                 qwerty::BasisVectorListAttr small,
                 std::deque<qwerty::BasisElemAttr> &big_queue,
                 std::deque<qwerty::BasisElemAttr> &small_queue,
                 llvm::SmallVectorImpl<qwerty::BasisElemAttr> &big_rebuilt,
                 llvm::SmallVectorImpl<qwerty::BasisElemAttr> &small_rebuilt) {
    assert(small.getDim() < big.getDim()
           && "Merge called when not needed?");
    size_t delta = big.getDim() - small.getDim();

    if (small_queue.empty()) {
        // No hope. There's nothing next.
        return false;
    }

    qwerty::BasisElemAttr next = small_queue.front();
    small_queue.pop_front();

    if (next.getDim() == delta) {
        big_rebuilt.push_back(rebuildZ(rewriter, big));
        small_rebuilt.push_back(rebuildZ(rewriter,
                                         merge(rewriter, small, next)));
        return true;
    } else if (next.getDim() < delta) {
        // continue, need to gobble more...
        qwerty::BasisVectorListAttr merged = merge(rewriter, small, next);
        return greedyMerge(rewriter, big, merged,
                           big_queue, small_queue,
                           big_rebuilt, small_rebuilt);
    } else { // next.getDim() > delta
        qwerty::BuiltinBasisAttr stdnext = next.getStd();
        qwerty::BasisVectorListAttr vlnext = next.getVeclist();

        if (stdnext) {
            qwerty::BasisVectorListAttr splat =
                splitStd(rewriter, small_queue, stdnext, delta)
                .expandToVeclist();
            big_rebuilt.push_back(rebuildZ(rewriter, big));
            small_rebuilt.push_back(rebuildZ(rewriter,
                                             merge(rewriter, small, splat)));
            return true;
        } else {  // vlnext
            qwerty::BasisVectorListAttr merged = merge(rewriter, small,
                                                       vlnext);
            // Swap roles to try and rebalance
            return greedyMerge(rewriter,
                               /*big=*/merged, /*small=*/big,
                               /*big_queue=*/small_queue,
                               /*small_queue=*/big_queue,
                               /*big_rebuilt=*/small_rebuilt,
                               /*small_rebuilt=*/big_rebuilt);
        }
    }
}

// A range of qubits along with the primitive basis it should be translated
// from/to. See the "standardization" subheading of Section 6.3 of the CGO
// paper.
struct Standardization {
    qwerty::PrimitiveBasis prim_basis;
    size_t start, end; // [start, end)
    bool unconditional;

    Standardization(qwerty::PrimitiveBasis prim_basis, size_t start, size_t end)
               : prim_basis(prim_basis), start(start), end(end), unconditional(false) {}

    Standardization(qwerty::PrimitiveBasis prim_basis, size_t start, size_t end, bool unconditional)
               : prim_basis(prim_basis), start(start), end(end), unconditional(unconditional) {}

    static Standardization new_padding(size_t start, size_t end) {
        return Standardization((qwerty::PrimitiveBasis)-1, start, end);
    }

    bool operator==(const Standardization &stdize) const {
        return prim_basis == stdize.prim_basis && stdize.start == start && stdize.end == end;
    }

    bool isPadding() const {
        return prim_basis == (qwerty::PrimitiveBasis)-1;
    }

    size_t get_dim() const {
        return end - start;
    }

    Standardization split(size_t new_end) {
        assert(new_end < end && "Standardization.split() called unnecessarily");
        size_t delta = end - new_end;
        end = new_end;
        return Standardization(prim_basis, new_end, new_end + delta);
    }
};

// Find standardization ranges in the list of basis elements provided, storing
// the results in stdize
void findStandardizations(
        llvm::SmallVectorImpl<Standardization> &stdize,
        llvm::ArrayRef<qwerty::BasisElemAttr> elems) {
    size_t qubit_idx = 0;
    for (qwerty::BasisElemAttr elem : elems) {
        // Little optimization: treat fourier[1] as pm. This will hopefully
        // help find more unconditional standardizations below
        qwerty::PrimitiveBasis prim_basis =
                elem.getPrimBasis() == qwerty::PrimitiveBasis::FOURIER
                && elem.getDim() == 1
            ? qwerty::PrimitiveBasis::X
            : elem.getPrimBasis();
        stdize.emplace_back(prim_basis, qubit_idx, qubit_idx + elem.getDim());
        qubit_idx += elem.getDim();
    }
}

bool primBasisIsSeparable(qwerty::PrimitiveBasis prim) {
    switch (prim) {
        case qwerty::PrimitiveBasis::X:
        case qwerty::PrimitiveBasis::Y:
        case qwerty::PrimitiveBasis::Z:
            return true;

        case qwerty::PrimitiveBasis::FOURIER:
        case qwerty::PrimitiveBasis::BELL:
            return false;

        default:
            assert(0 && "Primitive basis missing from primBasisIsSeparable()");
    }
}

// Implements Algorithm E6 of Appendix E of the CGO '25 paper. It updates
// left_stdize and right_stdize in-place to be correctly marked as conditional
// or unconditional.
void determineUnconditional(llvm::SmallVectorImpl<Standardization> &left_stdize,
                            llvm::SmallVectorImpl<Standardization> &right_stdize) {
    std::deque<Standardization> lqueue, rqueue;
    for (Standardization &std : left_stdize) {
        lqueue.push_back(std);
    }
    for (Standardization &std : right_stdize) {
        rqueue.push_back(std);
    }
    left_stdize.clear();
    right_stdize.clear();

    size_t pos = 0;
    while (!lqueue.empty() && !rqueue.empty()) {
        Standardization left = lqueue.front();
        lqueue.pop_front();
        Standardization right = rqueue.front();
        rqueue.pop_front();

        assert(left.start == right.start
               && "determineUnconditional() invariant violated: "
                  "standardizations do not start at the same position");

        bool unconditional = !left.isPadding() && !right.isPadding()
                             && left.prim_basis == right.prim_basis;
        size_t left_dim = left.get_dim();
        size_t right_dim = right.get_dim();
        if (left_dim == right_dim) {
            if (!left.isPadding()) {
                left_stdize.emplace_back(left.prim_basis, pos, pos + left_dim,
                                         unconditional);
            }
            if (!right.isPadding()) {
                right_stdize.emplace_back(right.prim_basis, pos,
                                          pos + right_dim, unconditional);
            }
            pos += left_dim;
        } else {
            size_t big_dim = std::max(left_dim, right_dim);
            size_t small_dim = std::min(left_dim, right_dim);

            Standardization &big = (left_dim > right_dim)? left : right;
            Standardization &small = (left_dim > right_dim)? right : left;

            llvm::SmallVectorImpl<Standardization> &big_stdize =
                (left_dim > right_dim)? left_stdize : right_stdize;
            llvm::SmallVectorImpl<Standardization> &small_stdize =
                (left_dim > right_dim)? right_stdize : left_stdize;

            std::deque<Standardization> &big_queue =
                (left_dim > right_dim) ? lqueue : rqueue;

            if (!big.isPadding() && primBasisIsSeparable(big.prim_basis)) {
                if (!small.isPadding()) {
                    small_stdize.emplace_back(small.prim_basis, pos,
                                              pos + small_dim, unconditional);
                }
                big_stdize.emplace_back(big.prim_basis, pos, pos + small_dim,
                                        unconditional);
                big_queue.push_front(Standardization(big.prim_basis,
                                                     pos + small_dim,
                                                     pos + big_dim));
            } else {
                if (!small.isPadding()) {
                    small_stdize.emplace_back(small.prim_basis, pos,
                                              pos + small_dim,
                                              /*unconditional=*/false);
                }
                if (!big.isPadding()) {
                    big_stdize.emplace_back(big.prim_basis, pos, pos + big_dim,
                                            /*unconditional=*/false);
                }
                big_queue.push_front(Standardization::new_padding(
                    pos + small_dim, pos + big_dim));
            }
            pos += small_dim;
        }
    }

    assert(lqueue.empty() && rqueue.empty()
           && "determineUnconditional() queues out of sync "
              "(invariants violated?)");
}

struct VectorPhase {
    size_t start; // qubit index
    llvm::APInt eigenbits;
    mlir::Value theta;

    VectorPhase(size_t start, llvm::APInt eigenbits, mlir::Value theta)
               : start(start), eigenbits(eigenbits), theta(theta) {}

    size_t getEnd() const {
        return start + eigenbits.getBitWidth();
    }
};

// Find the ranges of qubits and eigenbits upon the programmer requested us to
// (de)impart phases on particular vectors.
void findVectorPhases(llvm::SmallVectorImpl<VectorPhase> &vec_phases,
                      qwerty::BasisAttr basis,
                      mlir::ValueRange basis_phases,
                      size_t start_phase_idx) {
    size_t qubit_idx = 0;
    size_t phase_idx = start_phase_idx;
    for (qwerty::BasisElemAttr elem : basis.getElems()) {
        if (qwerty::BasisVectorListAttr veclist = elem.getVeclist()) {
            for (qwerty::BasisVectorAttr vec : veclist.getVectors()) {
                if (vec.hasPhase()) {
                    vec_phases.emplace_back(qubit_idx, vec.getEigenbits(),
                                            basis_phases[phase_idx++]);
                }
            }
        }
        qubit_idx += elem.getDim();
    }
}

// Rebuild a pair of bases such that they are both aligned (according to
// Section 6.3 of the CGO paper) and in the std/Z primitive basis.
mlir::LogicalResult rebuildAligned(mlir::Operation *offender,
                                   mlir::RewriterBase &rewriter,
                                   qwerty::BasisAttr basis_in,
                                   qwerty::BasisAttr basis_out,
                                   llvm::SmallVectorImpl<qwerty::BasisElemAttr> &left_rebuilt,
                                   llvm::SmallVectorImpl<qwerty::BasisElemAttr> &right_rebuilt) {
    std::deque<qwerty::BasisElemAttr> left_queue(basis_in.getElems().begin(),
                                                 basis_in.getElems().end());
    std::deque<qwerty::BasisElemAttr> right_queue(basis_out.getElems().begin(),
                                                  basis_out.getElems().end());

    while (!left_queue.empty() && !right_queue.empty()) {
        qwerty::BasisElemAttr left = left_queue.front();
        left_queue.pop_front();
        qwerty::BasisElemAttr right = right_queue.front();
        right_queue.pop_front();
        qwerty::BuiltinBasisAttr lstd = left.getStd();
        qwerty::BuiltinBasisAttr rstd = right.getStd();
        qwerty::BasisVectorListAttr lvl = left.getVeclist();
        qwerty::BasisVectorListAttr rvl = right.getVeclist();

        if (left.getDim() == right.getDim()) {
            if (lstd && rstd) {
                left_rebuilt.push_back(rebuildZ(rewriter, lstd));
                right_rebuilt.push_back(rebuildZ(rewriter, rstd));
            } else if (lstd) { // && rvl
                if (rvl.isPredicate()) {
                    return rewriter.notifyMatchFailure(offender,
                       "right veclist is not full. Span checking bug?");
                }

                left_rebuilt.push_back(rebuildZ(rewriter,
                                                lstd.expandToVeclist()));
                right_rebuilt.push_back(rebuildZ(rewriter, rvl));
            } else if (rstd) { // && lvl
                if (lvl.isPredicate()) {
                    return rewriter.notifyMatchFailure(offender,
                       "left veclist is not full. Span checking bug?");
                }

                left_rebuilt.push_back(rebuildZ(rewriter, lvl));
                right_rebuilt.push_back(rebuildZ(rewriter,
                                                 rstd.expandToVeclist()));
            } else { // lvl && rvl
                if ((lvl.isPredicate() || rvl.isPredicate())
                    && (lvl.getPrimBasis() != rvl.getPrimBasis()
                        || !isPermutation(lvl.getVectors(),
                                          rvl.getVectors()))) {
                    return rewriter.notifyMatchFailure(offender,
                       "Veclists are not permutations of one another. "
                       "How did this pass span checking?");
                }

                left_rebuilt.push_back(rebuildZ(rewriter, lvl));
                right_rebuilt.push_back(rebuildZ(rewriter, rvl));
            }
        } else {
            bool lbigger = left.getDim() > right.getDim();
            size_t small_dim = lbigger? right.getDim() : left.getDim();
            std::deque<qwerty::BasisElemAttr> &big_queue =
                lbigger? left_queue : right_queue;
            std::deque<qwerty::BasisElemAttr> &small_queue =
                lbigger? right_queue : left_queue;
            llvm::SmallVectorImpl<qwerty::BasisElemAttr> &big_rebuilt =
                lbigger? left_rebuilt : right_rebuilt;
            llvm::SmallVectorImpl<qwerty::BasisElemAttr> &small_rebuilt =
                lbigger? right_rebuilt : left_rebuilt;
            qwerty::BuiltinBasisAttr bigstd = lbigger? lstd : rstd;
            qwerty::BuiltinBasisAttr smallstd = lbigger? rstd : lstd;
            qwerty::BasisVectorListAttr bigvl = lbigger? lvl : rvl;
            qwerty::BasisVectorListAttr smallvl = lbigger? rvl : lvl;

            if (bigstd && smallstd) {
                big_rebuilt.push_back(rebuildZ(rewriter,
                                               splitStd(rewriter, big_queue,
                                                        bigstd, small_dim)));
                small_rebuilt.push_back(rebuildZ(rewriter, smallstd));
            } else if (bigstd && smallvl) {
                big_rebuilt.push_back(rebuildZ(rewriter,
                                               splitStd(rewriter, big_queue,
                                                        bigstd, small_dim)
                                               .expandToVeclist()));
                small_rebuilt.push_back(rebuildZ(rewriter, smallvl));
                if (smallvl.isPredicate()) {
                    return rewriter.notifyMatchFailure(offender,
                        "Small veclist is not full. Span checking broken?");
                }
            } else if (bigvl && smallstd) {
                if (qwerty::BasisElemAttr factored =
                        factorFull(rewriter, big_queue, bigvl, small_dim)) {
                    if (qwerty::BuiltinBasisAttr factored_std =
                            factored.getStd()) {
                        big_rebuilt.push_back(rebuildZ(rewriter,
                                                       factored_std));
                        small_rebuilt.push_back(rebuildZ(rewriter,
                                                         smallstd));
                    } else { // factored.getVeclist()
                        qwerty::BasisVectorListAttr factored_vl =
                            factored.getVeclist();
                        big_rebuilt.push_back(rebuildZ(rewriter,
                                                       factored_vl));
                        small_rebuilt.push_back(rebuildZ(rewriter,
                                                         smallstd
                                                         .expandToVeclist()));
                    }
                } else {
                    if (!greedyMerge(rewriter, bigvl,
                                     smallstd.expandToVeclist(),
                                     big_queue, small_queue,
                                     big_rebuilt, small_rebuilt)) {
                        return rewriter.notifyMatchFailure(offender,
                            "Merging failed");
                    }
                }
            } else { // bigvl && smallvl
                if (qwerty::BasisVectorListAttr factored =
                        factorVeclist(rewriter, big_queue, bigvl, smallvl)) {
                    big_rebuilt.push_back(rebuildZ(rewriter, factored));
                    small_rebuilt.push_back(rebuildZ(rewriter, smallvl));
                } else {
                    if (!greedyMerge(rewriter, bigvl, smallvl,
                                     big_queue, small_queue,
                                     big_rebuilt, small_rebuilt)) {
                        return rewriter.notifyMatchFailure(offender,
                            "Merging failed");
                    }
                }
            }
        }
    }

    if (!left_queue.empty() || !right_queue.empty()) {
        return rewriter.notifyMatchFailure(offender,
            "Invalid translation. How did this pass span checking?");
    }

    return mlir::success();
}

bool shouldSkipStandardization(const Standardization &stdize, bool unconditional) {
    return (stdize.unconditional ^ unconditional)
           || stdize.prim_basis == qwerty::PrimitiveBasis::Z;
}

// This is used below to narrow down the list of qubits to the ones that
// actually should be acted on in a (de)standardization. This dramatically
// simplifies the indexing arithmetic we need to do and leaves dealing with
// controls (if present) in the hands of lowerPredBasisToInterleavedControls().
void compactStandardizedQubits(llvm::SmallVectorImpl<Standardization> &standardizations,
                               bool unconditional,
                               llvm::SmallVectorImpl<mlir::Value> &qubits,
                               llvm::SmallVectorImpl<mlir::Value> &packed_out) {
    packed_out.clear();

    for (const Standardization &stdize : standardizations) {
        if (shouldSkipStandardization(stdize, unconditional)) {
            continue;
        }
        packed_out.append(qubits.begin() + stdize.start,
                          qubits.begin() + stdize.end);
    }
}

void uncompactStandardizedQubits(llvm::SmallVectorImpl<Standardization> &standardizations,
                                 bool unconditional,
                                 llvm::SmallVectorImpl<mlir::Value> &qubits_out,
                                 llvm::SmallVectorImpl<mlir::Value> &packed) {
    size_t qubit_idx = 0;
    for (const Standardization &stdize : standardizations) {
        if (shouldSkipStandardization(stdize, unconditional)) {
            continue;
        }
        for (size_t i = stdize.start; i < stdize.end; i++, qubit_idx++) {
            qubits_out[i] = packed[qubit_idx];
        }
    }
}

// Translate from other primitive bases to std and back again
void standardizeCompressed(mlir::RewriterBase &rewriter,
                           mlir::Location loc,
                           llvm::SmallVectorImpl<Standardization> &standardizations,
                           bool unconditional, bool left,
                           llvm::SmallVectorImpl<mlir::Value> &control_qubits,
                           // This is a "compacted" list produced by
                           // compactStandardizedQubits()
                           llvm::SmallVectorImpl<mlir::Value> &qubits) {
    auto one_qubit_gate = [&](qcirc::Gate1Q kind, ssize_t ctrl_idx, size_t idx) {
        qcirc::Gate1QOp gate;
        if (ctrl_idx >= 0) {
            llvm::SmallVector<mlir::Value> controls;
            controls.push_back(qubits[ctrl_idx]);
            controls.append(control_qubits.begin(), control_qubits.end());
            gate = rewriter.create<qcirc::Gate1QOp>(
                loc, kind, controls, qubits[idx]);
        } else {
            gate = rewriter.create<qcirc::Gate1QOp>(
                loc, kind, control_qubits, qubits[idx]);
        }
        qubits[idx] = gate.getResult();
        control_qubits.clear();
        if (ctrl_idx >= 0) {
            qubits[ctrl_idx] = gate.getControlResults()[0];
            control_qubits.append(gate.getControlResults().begin()+1,
                                  gate.getControlResults().end());
        } else {
            control_qubits.append(gate.getControlResults().begin(),
                                  gate.getControlResults().end());
        }
    };

    size_t qubit_idx = 0;
    for (const Standardization &stdize : standardizations) {
        if (shouldSkipStandardization(stdize, unconditional)) {
            continue;
        }

        size_t dim = stdize.end - stdize.start;

        if (stdize.prim_basis == qwerty::PrimitiveBasis::FOURIER) {
            if (left) {
                runInverseQft(loc, rewriter, control_qubits,
                              qubits, qubit_idx, dim);
            } else { // right
                runQft(loc, rewriter, control_qubits,
                       qubits, qubit_idx, dim);
            }
        } else if (stdize.prim_basis == qwerty::PrimitiveBasis::BELL) {
            assert(dim == 2 && "I only know the Bell basis on two qubits");
            if (left) {
                one_qubit_gate(qcirc::Gate1Q::X, 1, 0);
                one_qubit_gate(qcirc::Gate1Q::H, -1, 1);
                one_qubit_gate(qcirc::Gate1Q::Z, 1, 0);
            } else { // right
                one_qubit_gate(qcirc::Gate1Q::Z, 1, 0);
                one_qubit_gate(qcirc::Gate1Q::H, -1, 1);
                one_qubit_gate(qcirc::Gate1Q::X, 1, 0);
            }
        } else {
            for (size_t i = qubit_idx; i < qubit_idx + dim; i++) {
                switch (stdize.prim_basis) {
                case qwerty::PrimitiveBasis::X:
                    one_qubit_gate(qcirc::Gate1Q::H, -1, i);
                    break;
                case qwerty::PrimitiveBasis::Y:
                    if (left) {
                        one_qubit_gate(qcirc::Gate1Q::Sdg, -1, i);
                        one_qubit_gate(qcirc::Gate1Q::H, -1, i);
                    } else { // right
                        one_qubit_gate(qcirc::Gate1Q::H, -1, i);
                        one_qubit_gate(qcirc::Gate1Q::S, -1, i);
                    }
                    break;
                default:
                    assert(0 && "Missing handling for PrimitiveBasis in standardization in "
                                "basis translation lowering");
                }
            }
        }

        qubit_idx += dim;
    }
}

void standardizeCompressed(mlir::RewriterBase &rewriter,
                           mlir::Location loc,
                           llvm::SmallVectorImpl<Standardization> &standardizations,
                           bool unconditional, bool left,
                           // This is a "compacted" list produced by
                           // compactStandardizedQubits()
                           llvm::SmallVectorImpl<mlir::Value> &qubits) {
    llvm::SmallVector<mlir::Value> no_controls;
    standardizeCompressed(rewriter, loc, standardizations, unconditional, left,
                        no_controls, qubits);
}

// Outer layer of unconditional (de)standardizations. (See the
// "Standardization" subheading of Section 6.3 of the CGO paper.)
void standardizeUncond(mlir::RewriterBase &rewriter,
                       mlir::Location loc,
                       llvm::SmallVectorImpl<Standardization> &standardizations,
                       bool left,
                       llvm::SmallVectorImpl<mlir::Value> &qubits) {
    llvm::SmallVector<mlir::Value> stdize_qubits;
    compactStandardizedQubits(standardizations, /*unconditional=*/true,
                              qubits, stdize_qubits);
    standardizeCompressed(rewriter, loc, standardizations, /*unconditional=*/true, left,
                        stdize_qubits);
    uncompactStandardizedQubits(standardizations, /*unconditional=*/true,
                                qubits, stdize_qubits);
}

// Conditional (de)standardizations. (See the "Standardization" subheading of
// Section 6.3 of the CGO paper.)
void standardizeCond(mlir::RewriterBase &rewriter,
                     mlir::Location loc,
                     llvm::SmallVectorImpl<Standardization> &standardizations,
                     bool left,
                     llvm::SmallVectorImpl<mlir::Value> &qubits,
                     llvm::SmallVectorImpl<qwerty::BasisElemAttr> &rebuilt) {
    llvm::SmallVector<mlir::Value> cond_qubits;
    compactStandardizedQubits(standardizations, /*unconditional=*/false,
                              qubits, cond_qubits);
    size_t n_cond_qubits = cond_qubits.size();

    if (!n_cond_qubits) {
        return;
    }

    lowerPredBasisToInterleavedControls(
        rewriter, loc, rebuilt, qubits,
        [&](llvm::SmallVectorImpl<mlir::Value> &controls) {
            standardizeCompressed(rewriter, loc, standardizations,
                                  /*unconditional=*/false,
                                  left, controls, cond_qubits);
        });

    uncompactStandardizedQubits(standardizations, /*unconditional=*/false,
                                qubits, cond_qubits);
}

struct APIntCompare {
    bool operator()(const llvm::APInt &lhs, const llvm::APInt &rhs) const {
        return lhs.ult(rhs);
    }
};

// Remove a range of qubits from a vector list, de-duplicating the remainder on
// each side, and padding with std[N] in the range.
void chopOutRangeFromVeclist(
        mlir::Builder &builder,
        qwerty::BasisVectorListAttr vl,
        // Inclusive
        size_t chop_start,
        // Exclusive
        size_t chop_end,
        // Append chopped basis elements to this
        llvm::SmallVectorImpl<qwerty::BasisElemAttr> &elems_out) {
    llvm::SmallSet<llvm::APInt, 4, APIntCompare> left_seen, right_seen;
    size_t pad_dim = chop_end - chop_start;
    size_t dim = vl.getDim();
    for (qwerty::BasisVectorAttr vec : vl.getVectors()) {
        if (chop_start) {
            llvm::APInt left_eigenbits = vec.getEigenbits();
            left_eigenbits.lshrInPlace(dim - chop_start);
            left_eigenbits = left_eigenbits.trunc(chop_start);
            left_seen.insert(left_eigenbits);
        }

        if (chop_end < dim) {
            llvm::APInt right_eigenbits = vec.getEigenbits();
            right_eigenbits = right_eigenbits.trunc(dim - chop_end);
            right_seen.insert(right_eigenbits);
        }
    }

    if (!left_seen.empty()) {
        llvm::SmallVector<qwerty::BasisVectorAttr> left_vectors;
        for (const llvm::APInt &vec : left_seen) {
            left_vectors.push_back(builder.getAttr<qwerty::BasisVectorAttr>(
                qwerty::PrimitiveBasis::Z, vec, vec.getBitWidth(), false));
        }
        elems_out.push_back(builder.getAttr<qwerty::BasisElemAttr>(
            builder.getAttr<qwerty::BasisVectorListAttr>(left_vectors)));
    }

    elems_out.push_back(builder.getAttr<qwerty::BasisElemAttr>(
        builder.getAttr<qwerty::BuiltinBasisAttr>(
            qwerty::PrimitiveBasis::Z, pad_dim)));

    if (!right_seen.empty()) {
        llvm::SmallVector<qwerty::BasisVectorAttr> right_vectors;
        for (const llvm::APInt &vec : right_seen) {
            right_vectors.push_back(builder.getAttr<qwerty::BasisVectorAttr>(
                qwerty::PrimitiveBasis::Z, vec, vec.getBitWidth(), false));
        }
        elems_out.push_back(builder.getAttr<qwerty::BasisElemAttr>(
            builder.getAttr<qwerty::BasisVectorListAttr>(right_vectors)));
    }

}

// Chop a range of qubits out of a basis element, replacing it with a padding
// std[N]. (This ignores built-in bases since they can never be predicates and
// thus cannot be a concern.)
void chopOutRange(
        mlir::Builder &builder,
        qwerty::BasisElemAttr elem,
        // Inclusive
        size_t chop_start,
        // Exclusive
        size_t chop_end,
        // Append chopped basis elements to this
        llvm::SmallVectorImpl<qwerty::BasisElemAttr> &elems_out) {
    if (elem.getStd()) {
        // Doesn't even matter, not a predicate anyway
        elems_out.push_back(elem);
    } else if (qwerty::BasisVectorListAttr vl = elem.getVeclist()) {
        chopOutRangeFromVeclist(builder, vl, chop_start, chop_end, elems_out);
    } else {
        assert(0 && "Missing handling of basis element in chopOutVector()");
    }
}

// When imparting a vector phase, we need to predicate on the basis elements
// that are not part of the (phased) vector itself. Yet
// lowerPredBasisToInterleavedControls() still wants the basis we pass it to
// match the number of qubits. Complicating things further is that rebuilding
// the basis in an aligned from may have merged this vector with the phase into
// a larger basis literal, or even factored it into multiple smaller basis
// literals. The solution used here is to slice out the range of qubits
// corresponding to the (phased) vector from overlapping basis elements,
// replacing the sliced-out chunk with std[N].
qwerty::BasisAttr replaceVecWithPadding(
        mlir::Builder &builder,
        const VectorPhase &vp,
        llvm::SmallVectorImpl<qwerty::BasisElemAttr> &rebuilt) {
    llvm::SmallVector<qwerty::BasisElemAttr> pruned;

    size_t rebuilt_idx = 0;
    size_t qubit_idx = 0;
    while (rebuilt_idx < rebuilt.size()
           && qubit_idx + rebuilt[rebuilt_idx].getDim() <= vp.start) {
        qwerty::BasisElemAttr elem = rebuilt[rebuilt_idx];
        pruned.push_back(elem);
        qubit_idx += elem.getDim();
        rebuilt_idx++;
    }

    while (rebuilt_idx < rebuilt.size()
           && qubit_idx < vp.getEnd()) {
        qwerty::BasisElemAttr elem = rebuilt[rebuilt_idx];
        size_t dim = elem.getDim();

        size_t chop_start = vp.start < qubit_idx
                            ? 0
                            : vp.start - qubit_idx;
        size_t chop_end = qubit_idx+dim < vp.getEnd()
                          ? dim
                          : vp.getEnd() - qubit_idx;

        chopOutRange(builder, elem, chop_start, chop_end, pruned);

        qubit_idx += dim;
        rebuilt_idx++;
    }

    while (rebuilt_idx < rebuilt.size()) {
        qwerty::BasisElemAttr elem = rebuilt[rebuilt_idx];
        pruned.push_back(elem);
        qubit_idx += elem.getDim();
        rebuilt_idx++;
    }

    if (pruned.empty()) {
        return nullptr;
    } else {
        return builder.getAttr<qwerty::BasisAttr>(pruned);
    }
}

// Emit gates that impart (or un-impart if negate==true) vector phases for a
// basis translation. (For example, this code emits a relative phase gate for
// the basis translation {'1'} >> {-'1'}.)
void impartVecPhases(mlir::RewriterBase &rewriter,
                     mlir::Location loc,
                     llvm::SmallVectorImpl<VectorPhase> &phases,
                     bool negate,
                     llvm::SmallVectorImpl<mlir::Value> &qubits,
                     llvm::SmallVectorImpl<qwerty::BasisElemAttr> &rebuilt) {
    for (const VectorPhase &vp : phases) {
        // Conjugate with Xs. Conveniently, this works for both turning
        // 1-controls into 0-controls and also for using XP(theta)X for |0⟩
        // instead of P(theta) (for |1⟩)
        llvm::SmallVector<mlir::Value> vec_qubits(
            qubits.begin() + vp.start, qubits.begin() + vp.getEnd());
        for (size_t bit = 0; bit < vp.eigenbits.getBitWidth(); bit++) {
            size_t qubit_idx = vec_qubits.size()-1-bit;
            if (!vp.eigenbits[bit]) {
                vec_qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(loc,
                    qcirc::Gate1Q::X, mlir::ValueRange(),
                    vec_qubits[qubit_idx]).getResult();
            }
        }

        qwerty::BasisAttr pruned = replaceVecWithPadding(rewriter, vp, rebuilt);

        mlir::Value theta = vp.theta;
        if (negate) {
            theta = wrapStationaryFloatOps(
                rewriter, loc, theta,
                [&](mlir::ValueRange args) {
                    assert(args.size() == 1);
                    mlir::Value theta_arg = args[0];
                    return rewriter.create<mlir::arith::NegFOp>(loc, theta_arg).getResult();
                });
        }

        lowerPredBasisToInterleavedControls(
            rewriter, loc, pruned, qubits,
            [&](llvm::SmallVectorImpl<mlir::Value> &controls) {
                mlir::SmallVector<mlir::Value> p_controls(controls.begin(),
                                                          controls.end());
                p_controls.append(vec_qubits.begin(),
                                  vec_qubits.begin()+vec_qubits.size()-1);
                qcirc::Gate1Q1POp gate = rewriter.create<qcirc::Gate1Q1POp>(loc,
                    qcirc::Gate1Q1P::P, theta, p_controls,
                    vec_qubits[vec_qubits.size()-1]);
                size_t n_controls = controls.size();
                controls.clear();
                controls.append(gate.getControlResults().begin(),
                                gate.getControlResults().begin() + n_controls);
                vec_qubits.clear();
                vec_qubits.append(gate.getControlResults().begin() + n_controls,
                                  gate.getControlResults().end());
                vec_qubits.push_back(gate.getResult());
            });

        // Finish conjugation we started earlier
        for (size_t bit = 0; bit < vp.eigenbits.getBitWidth(); bit++) {
            size_t qubit_idx = vec_qubits.size()-1-bit;
            if (!vp.eigenbits[bit]) {
                vec_qubits[qubit_idx] = rewriter.create<qcirc::Gate1QOp>(loc,
                    qcirc::Gate1Q::X, mlir::ValueRange(),
                    vec_qubits[qubit_idx]).getResult();
            }
        }

        for (size_t i = vp.start; i < vp.getEnd(); i++) {
            qubits[i] = vec_qubits[i - vp.start];
        }
    }
}

// This is an optimization: if there are standardizations that are neither
// predicates nor involved in vector phases, there is no point performing them.
// For example, in the basis translation pm + std >> pm + ij, there is no point
// in synthesizing Hadamards to standardize and destandardize the first qubit.
// For {'p'} + std >> {'p'} + ij there would be, however, since {'p'} is a
// predicate.
void shootdownStandardizations(llvm::SmallVectorImpl<Standardization> &left_stdize,
                               llvm::SmallVectorImpl<Standardization> &right_stdize,
                               llvm::SmallVectorImpl<VectorPhase> &left_phases,
                               llvm::SmallVectorImpl<VectorPhase> &right_phases,
                               llvm::SmallVectorImpl<qwerty::BasisElemAttr> &left_rebuilt,
                               llvm::SmallVectorImpl<qwerty::BasisElemAttr> &right_rebuilt) {
    size_t left_stdize_idx = 0;
    size_t right_stdize_idx = 0;
    // For VectorPhases
    size_t left_phase_idx = 0;
    size_t right_phase_idx = 0;
    // For rebuilt
    size_t rebuilt_idx = 0;
    size_t qubit_idx = 0;

    for (;;) {
        // Skip past any conditional standardizations, those aren't interesting
        while (left_stdize_idx < left_stdize.size()
               && !left_stdize[left_stdize_idx].unconditional) {
            left_stdize_idx++;
        }
        while (right_stdize_idx < right_stdize.size()
               && !right_stdize[right_stdize_idx].unconditional) {
            right_stdize_idx++;
        }
        if (left_stdize_idx >= left_stdize.size()
                || right_stdize_idx >= right_stdize.size()) {
            break;
        }

        Standardization &lstdize = left_stdize[left_stdize_idx];
        Standardization &rstdize = right_stdize[right_stdize_idx];
        assert(lstdize == rstdize && "Corrupted standardize list");

        // First, try to determine if we'll apply some vector phase. If so, we
        // can't shootdown this standardization. This is an existential qualifier,
        // so just jump past earlier phases and check if the first phase is
        // past this standardization
        while (left_phase_idx < left_phases.size() &&
               left_phases[left_phase_idx].getEnd() <= lstdize.start) {
            left_phase_idx++;
        }
        while (right_phase_idx < right_phases.size() &&
               right_phases[right_phase_idx].getEnd() <= rstdize.start) {
            right_phase_idx++;
        }
        bool has_phase =
            (left_phase_idx < left_phases.size()
             && left_phases[left_phase_idx].start < lstdize.end)
            || (right_phase_idx < right_phases.size()
                && right_phases[right_phase_idx].start < rstdize.end);

        // Next, try to determine if we'll (later) do some binary permutation.
        // We need to check every rebuilt basis element that overlaps with this
        // standardization. This is conservative, but good enough
        // TODO: Be more precise. We could split separable basis to align with
        //       the rebuilt basis
        assert(left_rebuilt.size() == right_rebuilt.size()
               && "rebuilt are different sizes, can't possibly be aligned");
        size_t n_rebuilt = left_rebuilt.size();
        while (rebuilt_idx < n_rebuilt
               && qubit_idx < lstdize.start) { // lstdize.start == rstdize.start
            assert(left_rebuilt[rebuilt_idx].getDim() ==
                   right_rebuilt[rebuilt_idx].getDim()
                   && "Not aligned, dimension mismatch");
            qubit_idx += left_rebuilt[rebuilt_idx].getDim();
            rebuilt_idx++;
        }
        assert(rebuilt_idx < n_rebuilt
               && "Standardizations beyond end of rebuilt? How?");

        bool has_permut = false;
        bool has_pred = false;
        bool done = false;
        while (!done) {
            assert(!(left_rebuilt[rebuilt_idx].isPredicate()
                     ^ right_rebuilt[rebuilt_idx].isPredicate())
                   && "Invalid rebuilt basis, left xor right is a predicate");
            if (left_rebuilt[rebuilt_idx].isPredicate()) {
                has_pred = true;
            }
            if (left_rebuilt[rebuilt_idx] != right_rebuilt[rebuilt_idx]) {
                has_permut = true;
            }

            size_t rebuilt_end =
                qubit_idx + left_rebuilt[rebuilt_idx].getDim();
            if (rebuilt_end <= lstdize.end) {
                done = rebuilt_end == lstdize.end;
                assert(left_rebuilt[rebuilt_idx].getDim() ==
                       right_rebuilt[rebuilt_idx].getDim()
                       && "Not aligned, dimension mismatch");
                qubit_idx += left_rebuilt[rebuilt_idx].getDim();
                rebuilt_idx++;
            } else {
                // The next iteration will need to take a look at this guy,
                // don't increment it away yet
                done = true;
            }
        }

        if (has_phase || has_permut || has_pred) {
            // Cannot guarantee the safety of removing this standardization
        } else {
            // Shootdown! The easiest way to disable a standardization is setting
            // its PrimitiveBasis to Z
            lstdize.prim_basis = qwerty::PrimitiveBasis::Z;
            rstdize.prim_basis = qwerty::PrimitiveBasis::Z;
        }
        left_stdize_idx++;
        right_stdize_idx++;
    }
}

// This pattern does most of the work for synthesizing a basis translation
// besides synthesis of classical logic. Please see Section 6.3 of the CGO
// paper for details on the overarching ideas implemented here.
struct AlignBasisTranslations : public mlir::OpConversionPattern<qwerty::QBundleBasisTranslationOp> {
    using mlir::OpConversionPattern<qwerty::QBundleBasisTranslationOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleBasisTranslationOp trans,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basis_in = trans.getBasisIn();
        qwerty::BasisAttr basis_out = trans.getBasisOut();
        mlir::ValueRange basis_phases = trans.getBasisPhases();

        // If this basis translation is already aligned and in the standard
        // basis, we can skip this pattern entirely and let
        // SynthesizePermutations (below) continue with synthesis
        if (isAligned(basis_in, basis_out)) {
            return mlir::failure();
        }

        llvm::SmallVector<Standardization> left_stdize, right_stdize;
        findStandardizations(left_stdize, basis_in.getElems());
        findStandardizations(right_stdize, basis_out.getElems());
        determineUnconditional(left_stdize, right_stdize);

        llvm::SmallVector<VectorPhase> left_phases, right_phases;
        findVectorPhases(left_phases, basis_in, basis_phases, 0);
        findVectorPhases(right_phases, basis_out, basis_phases, basis_in.getNumPhases());

        llvm::SmallVector<qwerty::BasisElemAttr> left_rebuilt, right_rebuilt;
        mlir::LogicalResult ret = rebuildAligned(trans, rewriter, basis_in,
                                                 basis_out, left_rebuilt,
                                                 right_rebuilt);
        if (ret.failed()) {
            return ret;
        }

        // Must return success after this point

        // TODO: Do phase optimizations here

        shootdownStandardizations(left_stdize, right_stdize, left_phases, right_phases,
                                  left_rebuilt, right_rebuilt);

        mlir::Location loc = trans.getLoc();
        mlir::ValueRange unpacked = rewriter.create<qwerty::QBundleUnpackOp>(
            loc, trans.getQbundleIn()).getQubits();
        llvm::SmallVector<mlir::Value> qubits(unpacked.begin(),
                                              unpacked.end());

        // First, apply unconditional standardizations
        standardizeUncond(rewriter, loc, left_stdize, /*left=*/true, qubits);

        // Now apply conditional standardizations. Note that the span check in
        // type checking guarantees that predicates will always be unconditional.
        standardizeCond(rewriter, loc, left_stdize, /*left=*/true, qubits,
                        left_rebuilt);

        // Impart phases specified for individual vectors in the rhs
        impartVecPhases(rewriter, loc, left_phases, /*negate=*/true, qubits,
                        left_rebuilt);

        mlir::Value left_repacked = rewriter.create<qwerty::QBundlePackOp>(
            loc, qubits).getQbundle();
        qwerty::BasisAttr rebuilt_in =
            rewriter.getAttr<qwerty::BasisAttr>(left_rebuilt);
        qwerty::BasisAttr rebuilt_out =
            rewriter.getAttr<qwerty::BasisAttr>(right_rebuilt);
        mlir::Value after_rebuilt =
            rewriter.create<qwerty::QBundleBasisTranslationOp>(
                loc, rebuilt_in, rebuilt_out, mlir::ValueRange(),
                left_repacked).getQbundleOut();
        mlir::ValueRange after_rebuilt_unpacked =
            rewriter.create<qwerty::QBundleUnpackOp>(loc, after_rebuilt)
                    .getQubits();
        qubits = after_rebuilt_unpacked;

        // Impart phases specified for individual vectors in the rhs
        impartVecPhases(rewriter, loc, right_phases, /*negate=*/false, qubits,
                        right_rebuilt);

        // Apply conditional on rhs
        standardizeCond(rewriter, loc, right_stdize, /*left=*/false, qubits,
                        right_rebuilt);

        // Finally, undo unconditional rhs
        standardizeUncond(rewriter, loc, right_stdize, /*left=*/false, qubits);

        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(trans, qubits);
        return mlir::success();
    }
};

// Synthesize the reversible permutation at the heart of a basis translation
// circuit using the transformation-based synthesis algorithm from Tweedledum.
// This is expensive at compile time (exponentially expensive with respect to
// the number of qubits, actually) but produces a high-quality circuit.
void synthesizePermutationSlow(
        mlir::RewriterBase &rewriter,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx,
        qwerty::BasisVectorListAttr left,
        qwerty::BasisVectorListAttr right) {
    size_t dim = left.getDim();
    uint32_t two_to_the_n = 1ULL << dim;
    std::vector<uint32_t> perm(two_to_the_n);
    // Initialize this as identity. That way, we treat
    //     {'00','10'} >> {'10','00'}
    // as
    //     {'00','10','01','11'} >> {'10','00','01','11'}
    //                ^^^^^^^^^                ^^^^^^^^^
    for (uint32_t i = 0; i < perm.size(); i++) {
        perm[i] = i;
    }

    for (size_t i = 0; i < left.getVectors().size(); i++) {
        qwerty::BasisVectorAttr lv = left.getVectors()[i];
        qwerty::BasisVectorAttr rv = right.getVectors()[i];

        // reverseBits() is to account for Tweedledum using the opposite
        // endianness as we do
        uint32_t l_idx = static_cast<uint32_t>(
            lv.getEigenbits().reverseBits().getZExtValue());
        uint32_t r_idx = static_cast<uint32_t>(
            rv.getEigenbits().reverseBits().getZExtValue());

        perm[l_idx] = r_idx;
    }

    qcirc::synthPermutationSlow(rewriter, loc, control_qubits, qubits,
                                qubit_idx, perm);
}

// Synthesize the reversible permutation at the heart of a basis translation
// circuit using an algorithm that is basically the quantum equivalent of
// converting a classical truth table to a classical circuit of ANDs (for each
// row) and an OR (for each column). This is only used for large numbers of
// qubits where synthesizePermutationSlow() is too expensive.
void synthesizePermutationFast(
        mlir::RewriterBase &rewriter,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx,
        qwerty::BasisVectorListAttr left,
        qwerty::BasisVectorListAttr right) {
    llvm::SmallVector<std::pair<llvm::APInt, llvm::APInt>> perm;
    for (auto [left_vec, right_vec] : llvm::zip(left.getVectors(),
                                                right.getVectors())) {
        llvm::APInt left = left_vec.getEigenbits();
        llvm::APInt right = right_vec.getEigenbits();
        if (left != right) {
            perm.emplace_back(std::move(left), std::move(right));
        }
    }

    qcirc::synthPermutationFast(rewriter, loc, control_qubits, qubits,
                                qubit_idx, perm);
}

// Synthesize the reversible permutation of basis states at the heart of a
// basis translation circuit.
void synthesizePermutation(mlir::RewriterBase &rewriter,
                           mlir::Location loc,
                           llvm::SmallVectorImpl<mlir::Value> &control_qubits,
                           llvm::SmallVectorImpl<mlir::Value> &qubits,
                           size_t qubit_idx,
                           qwerty::BasisVectorListAttr left,
                           qwerty::BasisVectorListAttr right) {
    assert(left.getDim() == right.getDim()
           && left.getVectors().size() == right.getVectors().size()
           && "Not a permutation");
    size_t dim = left.getDim();

    // Run the O(2^n) algorithm (which generates a high-quality circuit) if n
    // is small. Else run a polynomial-time algorithm that emits a
    // lower-quality circuit.
    // How was 10 determined? I tried running the `megaperm` integration test
    // several times with different choices of thresholds, and it seems like 10
    // is crossover point after which our brute-force synthesis is faster.
    if (dim <= 10) {
        synthesizePermutationSlow(rewriter, loc, control_qubits, qubits,
                                  qubit_idx, left, right);
    } else {
        synthesizePermutationFast(rewriter, loc, control_qubits, qubits,
                                  qubit_idx, left, right);
    }
}

// This pattern matches only on basis translations that are aligned and in the
// standard basis. (That is, it synthesizes the permutation at the heart of a
// basis translation, as seen in Figure 6 of the CGO paper.) It is the job of
// AlignBasisTranslations (above) to produce basis translations that are
// aligned for this pattern to consume. See the "Permutation" paragraph in
// Section 6.3 of the CGO paper for details on this step.
struct SynthesizePermutations : public mlir::OpConversionPattern<qwerty::QBundleBasisTranslationOp> {
    using mlir::OpConversionPattern<qwerty::QBundleBasisTranslationOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleBasisTranslationOp trans,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basis_in = trans.getBasisIn();
        qwerty::BasisAttr basis_out = trans.getBasisOut();

        if (!isAligned(basis_in, basis_out)) {
            // AlignBasisTranslations needs to run first
            return mlir::failure();
        }

        // Must return success past this point

        mlir::Location loc = trans.getLoc();
        mlir::ValueRange unpacked =
            rewriter.create<qwerty::QBundleUnpackOp>(loc,
                trans.getQbundleIn()).getQubits();
        llvm::SmallVector<mlir::Value> qubits(unpacked.begin(),
                                              unpacked.end());

        assert(basis_in.getElems().size() == basis_out.getElems().size()
               && "bases not aligned. is isAligned() broken?");

        size_t qubit_idx = 0;
        for (size_t i = 0; i < basis_in.getElems().size(); i++) {
            qwerty::BasisElemAttr left = basis_in.getElems()[i];
            qwerty::BasisElemAttr right = basis_out.getElems()[i];

            if (left.getStd()) { // && right.getStd()
                assert(right.getStd() && "isAligned() busted");
                // Nothing to do, this is std[N] >> std[N]
            } else { // left.getVeclist() && right.getVeclist()
                qwerty::BasisVectorListAttr lvl = left.getVeclist();
                qwerty::BasisVectorListAttr rvl = right.getVeclist();
                assert(lvl && rvl && "isAligned() ain't working");

                if (lvl == rvl) {
                    // Nothing to do, thankfully, since this is just e.g.
                    // {'0','1'} >> {'0','1'} (an identity operation)
                } else {
                    size_t dim = left.getDim();

                    llvm::SmallVector<qwerty::BasisElemAttr> other_elems(
                        basis_in.getElems().begin(),
                        basis_in.getElems().begin() + i);
                    // Padding to obey the requirement of
                    // lowerPredBasisToInterleavedControls() that the dimension
                    // of the basis is the same as the size of the qubit array
                    other_elems.push_back(
                        rewriter.getAttr<qwerty::BasisElemAttr>(
                            rewriter.getAttr<qwerty::BuiltinBasisAttr>(
                                qwerty::PrimitiveBasis::Z, dim)));
                    other_elems.append(basis_in.getElems().begin() + i + 1,
                                       basis_in.getElems().end());

                    lowerPredBasisToInterleavedControls(
                        rewriter, loc, other_elems, qubits,
                        [&](llvm::SmallVectorImpl<mlir::Value> &controls) {
                            synthesizePermutation(rewriter, loc, controls,
                                                  qubits, qubit_idx, lvl, rvl);
                        });
                }
            }

            qubit_idx += left.getDim();
            assert(left.getDim() == right.getDim() && "isAligned() toast");
        }

        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(trans, qubits);
        return mlir::success();
    }
};

// Insert translations before non-std measurements
struct QBundleMeasureNonStd : public mlir::OpConversionPattern<qwerty::QBundleMeasureOp> {
    using mlir::OpConversionPattern<qwerty::QBundleMeasureOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleMeasureOp meas,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basisAttr = meas.getBasis();
        mlir::Value bundleVal = meas.getQbundle();

        if (basisAttr.hasPredicate()) {
            return mlir::failure();
        }

        llvm::ArrayRef<qwerty::BasisElemAttr> elems = basisAttr.getElems();
        // Avoid an infinite loop
        if (elems.size() == 1
                && elems[0].getStd()
                && elems[0].getStd().getPrimBasis() == qwerty::PrimitiveBasis::Z) {
            return mlir::failure();
        }

        uint64_t dim = basisAttr.getDim();
        qwerty::BuiltinBasisAttr sba1 = rewriter.getAttr<qwerty::BuiltinBasisAttr>(qwerty::PrimitiveBasis::Z,dim);
        qwerty::BasisElemAttr bea1 = rewriter.getAttr<qwerty::BasisElemAttr>(sba1);
        qwerty::BasisAttr ba1 = rewriter.getAttr<qwerty::BasisAttr>(bea1);

        qwerty::QBundleBasisTranslationOp btrans =
            rewriter.create<qwerty::QBundleBasisTranslationOp>(
                meas.getLoc(), basisAttr, ba1, mlir::ValueRange(), bundleVal);
        rewriter.replaceOpWithNewOp<qwerty::QBundleMeasureOp>(
            meas, ba1, btrans.getQbundleOut());
        return mlir::success();
    }
};

// Lowers only measurements in the std (Z or |0⟩/|1⟩) basis.
// QBundleMeasureNonStd (above) converts measurements in non-std bases to
// measurements in the standard basis, which this pattern then handles.
struct QBundleMeasureOpLowering : public mlir::OpConversionPattern<qwerty::QBundleMeasureOp> {
    using mlir::OpConversionPattern<qwerty::QBundleMeasureOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleMeasureOp meas,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basis = meas.getBasis();
        if (basis.getElems().size() != 1
                || !basis.getElems()[0].getStd()
                || basis.getElems()[0].getStd().getPrimBasis() != qwerty::PrimitiveBasis::Z) {
            // Let QBundleMeasureNonStd take care of this
            return mlir::failure();
        }

        // Now we can measure in the Z basis, since QBundleMeasureNonStd has
        // already converted the qubits to measure into that basis

        mlir::Location loc = meas.getLoc();
        uint64_t dim = meas.getQbundle().getType().getDim();
        mlir::ValueRange unpacked_range = rewriter.create<qwerty::QBundleUnpackOp>(loc, meas.getQbundle()).getQubits();
        llvm::SmallVector<mlir::Value> unpacked(unpacked_range);

        llvm::SmallVector<mlir::Value> bits(dim);
        for (uint64_t i = 0; i < dim; i++) {
            mlir::Value qubit = unpacked[i];
            qcirc::MeasureOp meas = rewriter.create<qcirc::MeasureOp>(loc, qubit);
            rewriter.create<qcirc::QfreeOp>(loc, meas.getQubitResult());
            bits[i] = meas.getMeasResult();
        }

        rewriter.replaceOpWithNewOp<qwerty::BitBundlePackOp>(meas, bits);
        return mlir::success();
    }
};

struct QBundleProjectNonStd : public mlir::OpConversionPattern<qwerty::QBundleProjectOp> {
    using mlir::OpConversionPattern<qwerty::QBundleProjectOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleProjectOp proj,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basisAttr = proj.getBasis();
        mlir::Value bundleVal = proj.getQbundleIn();

        if(basisAttr.hasPredicate()) {
            return mlir::failure();
        }

        llvm::ArrayRef<qwerty::BasisElemAttr> elems = basisAttr.getElems();
        // Avoid an infinite loop
        if(elems.size() == 1 && elems[0].getStd() && elems[0].getStd().getPrimBasis() == qwerty::PrimitiveBasis::Z) {
            return mlir::failure();
        }

        uint64_t dim = basisAttr.getDim();
        qwerty::BuiltinBasisAttr sba1 = rewriter.getAttr<qwerty::BuiltinBasisAttr>(qwerty::PrimitiveBasis::Z,dim);
        qwerty::BasisElemAttr bea1 = rewriter.getAttr<qwerty::BasisElemAttr>(sba1);
        qwerty::BasisAttr ba1 = rewriter.getAttr<qwerty::BasisAttr>(bea1);

        qwerty::QBundleBasisTranslationOp btrans =
            rewriter.create<qwerty::QBundleBasisTranslationOp>(
                proj.getLoc(), basisAttr, ba1, mlir::ValueRange(), bundleVal);

        qwerty::QBundleProjectOp proj2 =
            rewriter.create<qwerty::QBundleProjectOp>(
                proj.getLoc(), ba1, btrans.getQbundleOut());

        rewriter.replaceOpWithNewOp<qwerty::QBundleBasisTranslationOp>(
            proj, ba1, basisAttr, mlir::ValueRange(), proj2.getQbundleOut());
        return mlir::success();
    }
};

struct QBundleProjectOpLowering : public mlir::OpConversionPattern<qwerty::QBundleProjectOp> {
    using mlir::OpConversionPattern<qwerty::QBundleProjectOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleProjectOp proj,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basis = proj.getBasis();
        if (basis.getElems().size() != 1
                || !basis.getElems()[0].getStd()
                || basis.getElems()[0].getStd().getPrimBasis() != qwerty::PrimitiveBasis::Z) {
            // Let QBundleProjectNonStd take care of this
            return mlir::failure();
        }

        // Now we can measure in the Z basis, since QBundleProjectNonStd has
        // already converted the qubits to measure into that basis

        mlir::Location loc = proj.getLoc();
        uint64_t dim = proj.getQbundleIn().getType().getDim();
        mlir::ValueRange unpacked_range = rewriter.create<qwerty::QBundleUnpackOp>(loc, proj.getQbundleIn()).getQubits();
        llvm::SmallVector<mlir::Value> unpacked(unpacked_range);

        llvm::SmallVector<mlir::Value> measured;
        measured.reserve(dim);

        for (uint64_t i = 0; i < dim; i++) {
            mlir::Value qubit_in = unpacked[i];
            qcirc::MeasureOp meas = rewriter.create<qcirc::MeasureOp>(loc, qubit_in);
            measured.push_back(meas.getQubitResult());
        }

        rewriter.replaceOpWithNewOp<qwerty::QBundlePackOp>(proj, measured);
        return mlir::success();
    }
};

struct QBundleFlipOpLowering : public mlir::OpConversionPattern<qwerty::QBundleFlipOp> {
    using mlir::OpConversionPattern<qwerty::QBundleFlipOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleFlipOp flip,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        qwerty::BasisAttr basis = flip.getBasis();
        mlir::ValueRange basis_phases = flip.getBasisPhases();

        if (basis.getDim() != 1) {
            return rewriter.notifyMatchFailure(flip,
                    "Currently only 1D bases are supported in .flip");
        }
        if (basis.hasPredicate()) {
            return rewriter.notifyMatchFailure(flip,
                    "Predicates are not supported for .flip");
        }

        assert(basis.getElems().size() == 1 && "1D basis has more than 1 element, how?");
        qwerty::BasisElemAttr elem = basis.getElems()[0];
        qwerty::BasisVectorListAttr vector_list = elem.getVeclist();
        if (!vector_list) {
            assert(elem.getStd() && "Basis has neither built-in basis nor veclist!");
            vector_list = elem.getStd().expandToVeclist();
        }

        auto vectors = vector_list.getVectors();
        assert(vectors.size() == 2 && "Expected two vectors for 1D non-predicate");

        qwerty::BasisAttr rev_basis = rewriter.getAttr<qwerty::BasisAttr>(
            rewriter.getAttr<qwerty::BasisElemAttr>(
                rewriter.getAttr<qwerty::BasisVectorListAttr>(
                    std::initializer_list<qwerty::BasisVectorAttr>{vectors[1], vectors[0]})));

        llvm::SmallVector<mlir::Value> phases(basis_phases);
        auto rev_phases = llvm::reverse(basis_phases);
        phases.append(rev_phases.begin(), rev_phases.end());

        rewriter.replaceOpWithNewOp<qwerty::QBundleBasisTranslationOp>(
                flip, basis, rev_basis, phases, flip.getQbundleIn());
        return mlir::success();
    }
};

struct QBundleRotateOpLowering : public mlir::OpConversionPattern<qwerty::QBundleRotateOp> {
    using mlir::OpConversionPattern<qwerty::QBundleRotateOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleRotateOp rot,
                                        OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const final {
        mlir::Location loc = rot.getLoc();
        qwerty::BasisAttr basis = rot.getBasis();

        if (basis.getDim() != 1) {
            return rewriter.notifyMatchFailure(rot,
                    "Currently only 1D bases are supported in .rotate");
        }
        if (basis.hasPredicate()) {
            return rewriter.notifyMatchFailure(rot,
                    "Predicates are not supported for .rotate");
        }

        assert(basis.getElems().size() == 1 && "1D basis has more than 1 element, how?");
        qwerty::BasisElemAttr elem = basis.getElems()[0];
        qwerty::BasisVectorListAttr vector_list = elem.getVeclist();
        if (!vector_list) {
            assert(elem.getStd() && "Basis has neither built-in basis nor veclist!");
            vector_list = elem.getStd().expandToVeclist();
        }
        auto vectors = vector_list.getVectors();
        assert(vectors.size() == 2 && "Expected two vectors for 1D non-predicate");

        mlir::Value theta_by_2 = wrapStationaryFloatOps(
            rewriter, loc, rot.getTheta(),
            [&](mlir::ValueRange args) {
                assert(args.size() == 1);
                mlir::Value theta_arg = args[0];
                mlir::Value const_2 = rewriter.create<mlir::arith::ConstantOp>(
                        loc, rewriter.getF64FloatAttr(2.0)).getResult();
                return rewriter.create<mlir::arith::DivFOp>(
                    loc, theta_arg, const_2).getResult();
            });
        mlir::Value neg_theta_by_2 = wrapStationaryFloatOps(
            rewriter, loc, theta_by_2,
            [&](mlir::ValueRange args) {
                assert(args.size() == 1);
                mlir::Value theta_by_2_arg = args[0];
                return rewriter.create<mlir::arith::NegFOp>(
                    loc, theta_by_2_arg).getResult();
            });

        mlir::Value left_phase = neg_theta_by_2;
        mlir::Value right_phase = theta_by_2;

        qwerty::BasisAttr rot_basis = rewriter.getAttr<qwerty::BasisAttr>(
            rewriter.getAttr<qwerty::BasisElemAttr>(
                rewriter.getAttr<qwerty::BasisVectorListAttr>(
                    std::initializer_list<qwerty::BasisVectorAttr>{
                        rewriter.getAttr<qwerty::BasisVectorAttr>(
                            vectors[0].getPrimBasis(), vectors[0].getEigenbits(), vectors[0].getDim(), /*hasPhase=*/true),
                        rewriter.getAttr<qwerty::BasisVectorAttr>(
                            vectors[1].getPrimBasis(), vectors[1].getEigenbits(), vectors[1].getDim(), /*hasPhase=*/true)})));

        llvm::SmallVector<mlir::Value> phases{left_phase, right_phase};
        rewriter.replaceOpWithNewOp<qwerty::QBundleBasisTranslationOp>(
                rot, rot.getBasis(), rot_basis, phases, rot.getQbundleIn());
        return mlir::success();
    }
};

struct QwertyToQCircConversionPass : public qwerty::QwertyToQCircConversionBase<QwertyToQCircConversionPass> {
    void runOnOperation() override {
        mlir::ModuleOp module_op = getOperation();

        FuncUsages usages;
        if (mlir::failed(findFuncUsages(module_op, usages))) {
            llvm::errs() << "Func spec analysis failed";
            signalPassFailure();
            return;
        }

        QwertyToQCircTypeConverter type_converter(&getContext());
        if (mlir::failed(generateFuncSpecs(
                module_op, usages, type_converter))) {
            llvm::errs() << "Generating func specs failed";
            signalPassFailure();
            return;
        }

        mlir::ConversionTarget target(getContext());
        target.addIllegalDialect<qwerty::QwertyDialect>();
        target.addLegalDialect<mlir::BuiltinDialect,
                               qcirc::QCircDialect,
                               mlir::func::FuncDialect,
                               mlir::arith::ArithDialect,
                               mlir::scf::SCFDialect,
                               mlir::math::MathDialect>();
        // We have to hold the dialect conversion process's hand a little bit
        // for scf.if and qcirc.calc. We have to recreate them manually with
        // the right types
        target.addDynamicallyLegalOp<mlir::scf::IfOp,
                                     mlir::scf::YieldOp,
                                     mlir::arith::SelectOp,
                                     qcirc::CalcOp,
                                     qcirc::CalcYieldOp>(
            [&](mlir::Operation *op) {
                return type_converter.isLegal(op);
            });

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<SCFIfOpTypeFix,
                     SCFYieldOpTypeFix,
                     ArithSelectOpTypeFix,
                     CalcOpTypeFix,
                     CalcYieldOpTypeFix,
                     QBundlePackOpLowering,
                     QBundleUnpackOpLowering,
                     BitBundlePackOpLowering,
                     BitBundleUnpackOpLowering,
                     FuncOpLowering,
                     ReturnOpLowering,
                     CallOpLowering,
                     CallIndirectOpLowering,
                     FuncConstOpLowering,
                     FuncAdjointOpLowering,
                     FuncPredOpLowering,
                     BitInitOpLowering,
                     QBundleInitOpLowering,
                     QBundleDeinitOpLowering,
                     TrivialQBundlePrepOpLowering,
                     NontrivialQBundlePrepOpLowering,
                     QBundleDiscardOpLowering,
                     QBundleDiscardZeroOpLowering,
                     QBundlePhaseOpLowering,
                     QBundleIdentityOpLowering,
                     AlignBasisTranslations,
                     SynthesizePermutations,
                     QBundleMeasureNonStd,
                     QBundleMeasureOpLowering,
                     QBundleProjectNonStd,
                     QBundleProjectOpLowering,
                     QBundleFlipOpLowering,
                     QBundleRotateOpLowering,
                     SuperposOpLowering,
                     EnsembleOpLowering>(type_converter, &getContext());

        if (mlir::failed(mlir::applyFullConversion(module_op, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qwerty::createQwertyToQCircConversionPass() {
    return std::make_unique<QwertyToQCircConversionPass>();
}
