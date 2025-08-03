//===- QwertyOps.cpp - Qwerty dialect ops --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "QCirc/IR/QCircTypes.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/IR/QwertyDialect.h"

#include <unordered_set>

#define GET_OP_CLASSES
#include "Qwerty/IR/QwertyOps.cpp.inc"

namespace {
// General Steps to check if a qubit IR operation is linear with many uses:
// - Same block does not have two different use of an qubit IR instruction
// - Different blocks are allowed for control flow cases which means they cannot be dominating each other
bool linearCheckForManyUses(mlir::Value &value) {
    std::unordered_set<mlir::Block *> blocks;
    mlir::DominanceInfo domInfo;

    for(auto user : value.getUsers()) {
        mlir::Block *block = user->getBlock();

        for(auto oblock : blocks) {
            if(oblock == block ||
               domInfo.dominates(block, oblock) ||
               domInfo.dominates(oblock, block)) {
                return false;
            }
        }

        blocks.insert(block);
    }

    return true;
}

mlir::Value wrapStationaryFloatOps(mlir::RewriterBase &rewriter,
                                   mlir::Location loc,
                                   mlir::Value arg,
                                   std::function<mlir::Value(mlir::Value)> build_body) {
    qcirc::CalcOp calc = rewriter.create<qcirc::CalcOp>(loc, rewriter.getF64Type(), arg);
    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        // Sets insertion point to end of this block
        mlir::Block *calc_block = rewriter.createBlock(
            &calc.getRegion(), {}, arg.getType(),
            std::initializer_list<mlir::Location>{loc});
        assert(calc_block->getNumArguments() == 1);
        mlir::Value body_ret = build_body(calc_block->getArgument(0));
        rewriter.create<qcirc::CalcYieldOp>(loc, body_ret);
    }
    mlir::ValueRange calc_results = calc.getResults();
    assert(calc_results.size() == 1);
    return calc_results[0];
}

void buildPredicatedInit(
        // Parts of a QBundle{De,}InitOp
        mlir::Location loc,
        qwerty::BasisAttr basis,
        mlir::ValueRange basis_phases,
        mlir::Value qbundle_in,
        // true iff this is a QBundleDeinitOp
        bool reverse,
        // Normal parameters to buildPredicated()
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    size_t pred_dim = predBasis.getDim();

    mlir::ValueRange unpacked_pred = rewriter.create<qwerty::QBundleUnpackOp>(
        loc, predIn).getQubits();
    mlir::ValueRange unpacked_in = rewriter.create<qwerty::QBundleUnpackOp>(
        loc, qbundle_in).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_in.begin(), unpacked_in.end());
    mlir::Value repacked = rewriter.create<qwerty::QBundlePackOp>(
        loc, merged_qubits).getQbundle();

    // Convert pred & (theta1*'1' + theta2*'0').prep to the following
    // basis translation:
    // pred + std[N] >> pred + {theta1*'1','0'} + {theta2*'0','1'}
    llvm::SmallVector<qwerty::BasisElemAttr> lhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    lhs_elems.push_back(
        rewriter.getAttr<qwerty::BasisElemAttr>(
            rewriter.getAttr<qwerty::BuiltinBasisAttr>(
                qwerty::PrimitiveBasis::Z, basis.getDim())));

    llvm::SmallVector<qwerty::BasisElemAttr> rhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());

    for (qwerty::BasisElemAttr elem : basis.getElems()) {
        qwerty::BasisVectorListAttr veclist =
            llvm::cast<qwerty::BasisVectorListAttr>(elem.getVeclist());
        for (qwerty::BasisVectorAttr vec : veclist.getVectors()) {
            for (size_t i = 0; i < vec.getDim(); i++) {
                bool hasPhase = !i && vec.hasPhase();
                bool bit = vec.getEigenbits()[vec.getDim()-1-i];
                qwerty::BasisVectorAttr zero =
                    rewriter.getAttr<qwerty::BasisVectorAttr>(
                        vec.getPrimBasis(), qwerty::Eigenstate::PLUS,
                        /*dim=*/1, /*hasPhase=*/!bit && hasPhase);
                qwerty::BasisVectorAttr one =
                    rewriter.getAttr<qwerty::BasisVectorAttr>(
                        vec.getPrimBasis(), qwerty::Eigenstate::MINUS,
                        /*dim=*/1, /*hasPhase=*/bit && hasPhase);
                rhs_elems.push_back(
                    rewriter.getAttr<qwerty::BasisElemAttr>(
                        rewriter.getAttr<qwerty::BasisVectorListAttr>(
                            bit? std::initializer_list<qwerty::BasisVectorAttr>
                                 {one, zero}
                               : std::initializer_list<qwerty::BasisVectorAttr>
                                 {zero, one})));
            }
        }
    }

    mlir::Value res = rewriter.create<qwerty::QBundleBasisTranslationOp>(loc,
        rewriter.getAttr<qwerty::BasisAttr>(reverse? rhs_elems : lhs_elems),
        rewriter.getAttr<qwerty::BasisAttr>(reverse? lhs_elems : rhs_elems),
        basis_phases,
        repacked).getQbundleOut();

    mlir::ValueRange res_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(
        loc, res).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        res_unpacked.begin(), res_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        res_unpacked.begin() + pred_dim, res_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<qwerty::QBundlePackOp>(
        loc, pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<qwerty::QBundlePackOp>(
        loc, res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

struct FuncConstBinder {
    bool &bind_adj;
    qwerty::BasisAttr &bind_pred;
    mlir::FlatSymbolRefAttr &bind_symbol;
    mlir::ValueRange &bind_captures;

    FuncConstBinder(bool &bind_adj,
                    qwerty::BasisAttr &bind_pred,
                    mlir::FlatSymbolRefAttr &bind_symbol,
                    mlir::ValueRange &bind_captures)
                   : bind_adj(bind_adj),
                     bind_pred(bind_pred),
                     bind_symbol(bind_symbol),
                     bind_captures(bind_captures) {}

    bool match(mlir::Operation *op) {
        bind_adj = false;
        bind_pred = nullptr;

        qwerty::FuncAdjointOp func_adj;
        qwerty::FuncPredOp func_pred;
        // Why _or_null here? Below we set op = (some Value).getDefiningOp().
        // If the Value is actually a BlockArgument, then op will be null.
        while ((func_adj = llvm::dyn_cast_or_null<qwerty::FuncAdjointOp>(op))
                || (func_pred = llvm::dyn_cast_or_null<qwerty::FuncPredOp>(op))) {
            if (func_adj) {
                bind_adj = !bind_adj;
                op = func_adj.getCallee().getDefiningOp();
            } else { // func_pred
                qwerty::BasisAttr this_pred = func_pred.getPred();
                if (!bind_pred) {
                    bind_pred = this_pred;
                } else {
                    // Tensor product
                    llvm::SmallVector<qwerty::BasisElemAttr> elems(
                        this_pred.getElems().begin(),
                        this_pred.getElems().end());
                    elems.append(bind_pred.getElems().begin(),
                                 bind_pred.getElems().end());
                    bind_pred = qwerty::BasisAttr::get(
                        op->getContext(), elems);
                }
                op = func_pred.getCallee().getDefiningOp();
            }
        }

        qwerty::FuncConstOp func_const =
            llvm::dyn_cast_or_null<qwerty::FuncConstOp>(op);
        if (!func_const) {
            return false;
        }
        bind_symbol = func_const.getFuncAttr();
        bind_captures = func_const.getCaptures();

        return true;
    }
};

FuncConstBinder m_FuncConst(bool &bind_adj,
                            qwerty::BasisAttr &bind_pred,
                            mlir::FlatSymbolRefAttr &bind_symbol,
                            mlir::ValueRange &bind_captures) {
    return FuncConstBinder(bind_adj, bind_pred, bind_symbol, bind_captures);
}

struct CallIndirectConst : public mlir::OpRewritePattern<qwerty::CallIndirectOp> {
    using OpRewritePattern<qwerty::CallIndirectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::CallIndirectOp call,
                                        mlir::PatternRewriter &rewriter) const override {
        bool adj;
        qwerty::BasisAttr pred;
        mlir::FlatSymbolRefAttr symbol;
        mlir::ValueRange captures;

        mlir::Operation *upstream = call.getCallee().getDefiningOp();
        if (!upstream || !mlir::matchPattern(upstream,
                                             m_FuncConst(adj, pred, symbol,
                                                         captures))) {
            return mlir::failure();
        }

        llvm::SmallVector<mlir::Value> captures_and_operands(captures);
        captures_and_operands.append(call.getCallOperands().begin(),
                                     call.getCallOperands().end());

        rewriter.replaceOpWithNewOp<qwerty::CallOp>(call,
            call.getResultTypes(), symbol, /*adj=*/adj, /*pred=*/pred,
            captures_and_operands);
        return mlir::success();
    }
};

bool block_is_duplicatable(mlir::Block &in_block) {
    for (mlir::Operation &op : in_block) {
        DEBUG_WITH_TYPE("duplicatable", llvm::errs() << "Checking op");
        if (!mlir::isPure(&op)) {
            DEBUG_WITH_TYPE("duplicatable", {
                llvm::errs() << "Op is not Pure";
                op.dump();
            });
            return false;
        }
        for (mlir::Value operand : op.getOperands()) {
            if (llvm::isa<qcirc::NonStationaryTypeInterface>(operand.getType()) && operand.getParentBlock() != &in_block) {
                DEBUG_WITH_TYPE("duplicatable", {
                    llvm::errs() << "Operand is not defined in the same scope as block";
                    op.dump();
                    operand.dump();
                });
                return false;
            }
        }
    }
    return true;
}

struct CallIndirectIf : public mlir::OpRewritePattern<qwerty::CallIndirectOp> {
    using OpRewritePattern<qwerty::CallIndirectOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::CallIndirectOp call,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::scf::IfOp if_op =
            call.getCallee().getDefiningOp<mlir::scf::IfOp>();
        //TODO: generalize this pattern to include ifOps returning more than 1 value
        if (!if_op || if_op.getResults().size() != 1) {
            return mlir::failure();
        }

        if (!block_is_duplicatable(*if_op.thenBlock())) {
            return mlir::failure();
        }
        if (!block_is_duplicatable(*if_op.elseBlock())) {
            return mlir::failure();
        }

        rewriter.setInsertionPoint(call);
        mlir::scf::IfOp new_if_op =
            rewriter.create<mlir::scf::IfOp>(
                if_op.getLoc(), call.getResultTypes(), if_op.getCondition());
        rewriter.cloneRegionBefore(if_op.getThenRegion(),
                                   new_if_op.getThenRegion(),
                                   new_if_op.getThenRegion().end());
        rewriter.cloneRegionBefore(if_op.getElseRegion(),
                                   new_if_op.getElseRegion(),
                                   new_if_op.getElseRegion().end());

        mlir::scf::YieldOp then_yield = new_if_op.thenYield();
        mlir::scf::YieldOp else_yield = new_if_op.elseYield();
        assert(then_yield && else_yield
               // If the scf.if returned 1 Value, then the yields must have 1
               // operand too
               && then_yield.getResults().size() == 1
               && else_yield.getResults().size() == 1);

        rewriter.setInsertionPoint(then_yield);
        mlir::Value then_callee = then_yield.getResults()[0];
        mlir::ValueRange then_call_results =
            rewriter.create<qwerty::CallIndirectOp>(
                call.getLoc(), then_callee,
                call.getCallOperands()).getResults();
        rewriter.create<mlir::scf::YieldOp>(then_yield.getLoc(),
                                            then_call_results);
        rewriter.eraseOp(then_yield);

        rewriter.setInsertionPoint(else_yield);
        mlir::Value else_callee = else_yield.getResults()[0];
        mlir::ValueRange else_call_results =
            rewriter.create<qwerty::CallIndirectOp>(
                call.getLoc(), else_callee,
                call.getCallOperands()).getResults();
        rewriter.create<mlir::scf::YieldOp>(else_yield.getLoc(),
                                            else_call_results);
        rewriter.eraseOp(else_yield);

        rewriter.replaceOp(call, new_if_op.getResults());
        if (if_op.getResults()[0].use_empty()) {
            rewriter.eraseOp(if_op);
        } else {
            mlir::Operation *tgt = nullptr;
            for (mlir::Operation *user : if_op->getUsers()) {
                mlir::Operation *op = user;
                while (op->getBlock() != if_op->getBlock()) {
                    op = op->getParentOp();
                    assert(op && "Cannot move scf.if because downstream op has no parent in this block");
                } 
                if (!tgt) {
                    tgt = op;
                } else if (op->isBeforeInBlock(tgt)) {
                    tgt = op;
                }
            }
            assert(tgt && "No users of scf.if found, cannot move it");
            rewriter.moveOpBefore(if_op, tgt);
        }
        return mlir::success();
    }
};

struct AdjAdj : public mlir::OpRewritePattern<qwerty::FuncAdjointOp> {
    using OpRewritePattern<qwerty::FuncAdjointOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncAdjointOp func_adj_op,
                                        mlir::PatternRewriter &rewriter) const override {
        qwerty::FuncAdjointOp upstream_adj =
            func_adj_op.getCallee().getDefiningOp<qwerty::FuncAdjointOp>();

        if (!upstream_adj) {
            return mlir::failure();
        }

        rewriter.replaceOp(func_adj_op, upstream_adj.getCallee());
        return mlir::success();
    }
};

struct PredPred : public mlir::OpRewritePattern<qwerty::FuncPredOp> {
    using OpRewritePattern<qwerty::FuncPredOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncPredOp pred,
                                        mlir::PatternRewriter &rewriter) const override {
        qwerty::FuncPredOp upstream_pred =
            pred.getCallee().getDefiningOp<qwerty::FuncPredOp>();
        if (!upstream_pred) {
            return mlir::failure();
        }

        llvm::SmallVector<qwerty::BasisElemAttr> new_elems(
            pred.getPred().getElems());
        new_elems.append(upstream_pred.getPred().getElems().begin(),
                         upstream_pred.getPred().getElems().end());
        qwerty::BasisAttr new_basis =
            rewriter.getAttr<qwerty::BasisAttr>(new_elems);
        assert(new_basis.getDim() ==
               upstream_pred.getPred().getDim() + pred.getPred().getDim());

        rewriter.replaceOpWithNewOp<qwerty::FuncPredOp>(
            pred, new_basis, upstream_pred.getCallee());
        return mlir::success();
    }
};

struct PointlessPred : public mlir::OpRewritePattern<qwerty::FuncPredOp> {
    using OpRewritePattern<qwerty::FuncPredOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncPredOp pred,
                                        mlir::PatternRewriter &rewriter) const override {
        qwerty::BasisAttr basis = pred.getPred();
        if (!basis.hasNonPredicate()) {
            return mlir::failure();
        }

        mlir::Location loc = pred.getLoc();
        qwerty::FunctionType func_type = pred.getResult().getType();
        mlir::FunctionType inner_func_type = func_type.getFunctionType();
        qwerty::LambdaOp lambda =
            rewriter.create<qwerty::LambdaOp>(
                loc, func_type, pred.getCallee());
        mlir::Region &lambda_region = lambda.getRegion();

        // We assume this for every reversible function (and this must be a
        // reversible function if we are predicating it)
        assert(inner_func_type.getNumInputs() == 1
               && inner_func_type.getNumResults() == 1
               && inner_func_type.getInputs() == inner_func_type.getResults()
               && llvm::isa<qwerty::QBundleType>(
                      inner_func_type.getInputs()[0]));
        qwerty::QBundleType qbundle_type =
            llvm::cast<qwerty::QBundleType>(inner_func_type.getInputs()[0]);

        {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            // Sets insertion point to start of this block
            mlir::Block *block =
                rewriter.createBlock(
                    &lambda_region,
                    {},
                    {pred.getCallee().getType(), qbundle_type},
                    {loc, loc});
            assert(block->getNumArguments() == 2);
            mlir::Value og_func = block->getArgument(0);
            mlir::Value qbundle_in = block->getArgument(1);
            mlir::ValueRange qubits_in =
                rewriter.create<qwerty::QBundleUnpackOp>(
                    loc, qbundle_in).getQubits();

            llvm::SmallVector<mlir::Value> func_args;
            llvm::SmallVector<qwerty::BasisElemAttr> new_elems;

            size_t qubit_idx = 0;
            for (qwerty::BasisElemAttr elem : basis.getElems()) {
                if (elem.isPredicate()) {
                    new_elems.push_back(elem);
                    func_args.append(
                        qubits_in.begin() + qubit_idx,
                        qubits_in.begin() + qubit_idx + elem.getDim());
                }
                qubit_idx += elem.getDim();
            }
            assert(qubit_idx == basis.getDim());
            // Actual arguments to the function
            func_args.append(qubits_in.begin() + qubit_idx, qubits_in.end());

            qwerty::BasisAttr new_basis;
            if (!new_elems.empty()) {
                new_basis = rewriter.getAttr<qwerty::BasisAttr>(new_elems);
            }

            mlir::Value func_args_packed =
                rewriter.create<qwerty::QBundlePackOp>(
                    loc, func_args).getQbundle();
            mlir::Value new_func = og_func;
            if (new_basis) {
                new_func = rewriter.create<qwerty::FuncPredOp>(
                    loc, new_basis, new_func).getResult();
            }
            mlir::ValueRange func_results =
                rewriter.create<qwerty::CallIndirectOp>(
                    loc, new_func, func_args_packed).getResults();
            assert(func_results.size() == 1);
            mlir::Value func_result = func_results[0];
            mlir::ValueRange result_qubits =
                rewriter.create<qwerty::QBundleUnpackOp>(
                    loc, func_result).getQubits();

            llvm::SmallVector<mlir::Value> final_qubits;
            qubit_idx = 0;
            size_t result_idx = 0;
            for (qwerty::BasisElemAttr elem : basis.getElems()) {
                size_t elem_dim = elem.getDim();
                if (elem.isPredicate()) {
                    final_qubits.append(
                        result_qubits.begin() + result_idx,
                        result_qubits.begin() + result_idx + elem_dim);
                    result_idx += elem_dim;
                } else {
                    final_qubits.append(
                        qubits_in.begin() + qubit_idx,
                        qubits_in.begin() + qubit_idx + elem_dim);
                }
                qubit_idx += elem_dim;
            }
            assert(qubit_idx == basis.getDim()
                   && ((!new_basis && !result_idx)
                       || (new_basis && result_idx == new_basis.getDim())));
            final_qubits.append(result_qubits.begin() + result_idx,
                                result_qubits.end());

            mlir::Value final_packed =
                rewriter.create<qwerty::QBundlePackOp>(
                    loc, final_qubits).getQbundle();
            rewriter.create<qwerty::ReturnOp>(loc, final_packed);
        }

        rewriter.replaceOp(pred, lambda.getResult());

        return mlir::success();
    }
};

struct CallWithPointlessPred : public mlir::OpRewritePattern<qwerty::CallOp> {
    using OpRewritePattern<qwerty::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::CallOp call,
                                        mlir::PatternRewriter &rewriter) const override {
        if (!call.getPred()) {
            return mlir::failure();
        }
        qwerty::BasisAttr basis = call.getPredAttr();
        if (!basis.hasNonPredicate()) {
            return mlir::failure();
        }

        mlir::Location loc = call.getLoc();
        mlir::ValueRange call_operands = call.getCapturesAndOperands();
        // Wouldn't make any sense to predicate something with the mandatory
        // qbundle input
        assert(!call_operands.empty());
        mlir::Value qbundle_in = call_operands[call_operands.size()-1];
        mlir::ValueRange qubits_in =
            rewriter.create<qwerty::QBundleUnpackOp>(
                loc, qbundle_in).getQubits();

        llvm::SmallVector<mlir::Value> func_args;
        llvm::SmallVector<qwerty::BasisElemAttr> new_elems;
        size_t qubit_idx = 0;
        for (qwerty::BasisElemAttr elem : basis.getElems()) {
            size_t elem_dim = elem.getDim();
            if (elem.isPredicate()) {
                new_elems.push_back(elem);
                func_args.append(
                    qubits_in.begin() + qubit_idx,
                    qubits_in.begin() + qubit_idx + elem_dim);
            }
            qubit_idx += elem_dim;
        }
        // Actual arguments to the function
        func_args.append(qubits_in.begin() + qubit_idx, qubits_in.end());

        size_t new_n_qubits = func_args.size();
        qwerty::QBundleType func_arg_type =
            rewriter.getType<qwerty::QBundleType>(new_n_qubits);

        qwerty::BasisAttr new_basis;
        if (!new_elems.empty()) {
            new_basis = rewriter.getAttr<qwerty::BasisAttr>(new_elems);
        }

        mlir::Value func_args_packed =
            rewriter.create<qwerty::QBundlePackOp>(
                loc, func_args).getQbundle();
        // Keep whatever captures we were passing before
        llvm::SmallVector<mlir::Value> new_call_operands(
            call_operands.begin(),
            call_operands.begin()+(call_operands.size()-1));
        new_call_operands.push_back(func_args_packed);

        mlir::ValueRange func_results =
            rewriter.create<qwerty::CallOp>(
                loc, func_arg_type, call.getCalleeAttr(), call.getAdj(),
                new_basis, new_call_operands).getResults();
        assert(func_results.size() == 1);
        mlir::Value func_result = func_results[0];
        mlir::ValueRange result_qubits =
            rewriter.create<qwerty::QBundleUnpackOp>(
                loc, func_result).getQubits();

        llvm::SmallVector<mlir::Value> final_qubits;
        qubit_idx = 0;
        size_t result_idx = 0;
        for (qwerty::BasisElemAttr elem : basis.getElems()) {
            size_t elem_dim = elem.getDim();
            if (elem.isPredicate()) {
                final_qubits.append(
                    result_qubits.begin() + result_idx,
                    result_qubits.begin() + result_idx + elem_dim);
                result_idx += elem_dim;
            } else {
                final_qubits.append(
                    qubits_in.begin() + qubit_idx,
                    qubits_in.begin() + qubit_idx + elem_dim);
            }
            qubit_idx += elem_dim;
        }
        assert(qubit_idx == basis.getDim()
               && ((!new_basis && !result_idx)
                   || (new_basis && result_idx == new_basis.getDim())));
        final_qubits.append(result_qubits.begin() + result_idx,
                            result_qubits.end());

        mlir::Value final_packed =
            rewriter.create<qwerty::QBundlePackOp>(
                loc, final_qubits).getQbundle();
        rewriter.replaceOp(call, final_packed);
        return mlir::success();
    }
};

struct AdjIf : public mlir::OpRewritePattern<qwerty::FuncAdjointOp> {
    using OpRewritePattern<qwerty::FuncAdjointOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncAdjointOp func_adj_op,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::scf::IfOp if_op =
            func_adj_op.getCallee().getDefiningOp<mlir::scf::IfOp>();
        if (!if_op
                // If there are many results used all over the place, we can't
                // safely repurpose it as spitting out the result of the
                // func_adj
                || if_op.getResults().size() != 1
                // If there is someone else using the function this is
                // returning, we can't safely get rid of this scf.if
                || !if_op.getResults()[0].hasOneUse()) {
            return mlir::failure();
        }

        rewriter.setInsertionPoint(if_op);
        mlir::scf::IfOp new_if_op =
            rewriter.create<mlir::scf::IfOp>(
                if_op.getLoc(), func_adj_op.getResult().getType(),
                if_op.getCondition());
        rewriter.inlineRegionBefore(if_op.getThenRegion(),
                                    new_if_op.getThenRegion(),
                                    new_if_op.getThenRegion().end());
        rewriter.inlineRegionBefore(if_op.getElseRegion(),
                                    new_if_op.getElseRegion(),
                                    new_if_op.getElseRegion().end());

        mlir::scf::YieldOp then_yield = new_if_op.thenYield();
        mlir::scf::YieldOp else_yield = new_if_op.elseYield();
        assert(then_yield && else_yield
               // If the scf.if returned 1 Value, then the yields must have 1
               // operand too
               && then_yield.getResults().size() == 1
               && else_yield.getResults().size() == 1);

        rewriter.setInsertionPoint(then_yield);
        mlir::Value then_func = then_yield.getResults()[0];
        mlir::Value then_adj =
            rewriter.create<qwerty::FuncAdjointOp>(
                func_adj_op.getLoc(), then_func).getResult();
        rewriter.create<mlir::scf::YieldOp>(then_yield.getLoc(),
                                            then_adj);
        rewriter.eraseOp(then_yield);

        rewriter.setInsertionPoint(else_yield);
        mlir::Value else_func = else_yield.getResults()[0];
        mlir::Value else_adj =
            rewriter.create<qwerty::FuncAdjointOp>(
                func_adj_op.getLoc(), else_func).getResult();
        rewriter.create<mlir::scf::YieldOp>(else_yield.getLoc(),
                                            else_adj);
        rewriter.eraseOp(else_yield);

        rewriter.replaceOp(func_adj_op, new_if_op.getResults());
        // Now that the old scf.if has no uses, we can finally erase it
        rewriter.eraseOp(if_op);
        return mlir::success();
    }
};

struct PredIf : public mlir::OpRewritePattern<qwerty::FuncPredOp> {
    using OpRewritePattern<qwerty::FuncPredOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::FuncPredOp func_pred_op,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::scf::IfOp if_op =
            func_pred_op.getCallee().getDefiningOp<mlir::scf::IfOp>();
        if (!if_op
                // If there are many results used all over the place, we can't
                // safely repurpose it as spitting out the result of the
                // func_pred
                || if_op.getResults().size() != 1
                // If there is someone else using the function this is
                // returning, we can't safely get rid of this scf.if
                || !if_op.getResults()[0].hasOneUse()) {
            return mlir::failure();
        }

        rewriter.setInsertionPoint(if_op);
        mlir::scf::IfOp new_if_op =
            rewriter.create<mlir::scf::IfOp>(
                if_op.getLoc(), func_pred_op.getResult().getType(),
                if_op.getCondition());
        rewriter.inlineRegionBefore(if_op.getThenRegion(),
                                    new_if_op.getThenRegion(),
                                    new_if_op.getThenRegion().end());
        rewriter.inlineRegionBefore(if_op.getElseRegion(),
                                    new_if_op.getElseRegion(),
                                    new_if_op.getElseRegion().end());

        mlir::scf::YieldOp then_yield = new_if_op.thenYield();
        mlir::scf::YieldOp else_yield = new_if_op.elseYield();
        assert(then_yield && else_yield
               // If the scf.if returned 1 Value, then the yields must have 1
               // operand too
               && then_yield.getResults().size() == 1
               && else_yield.getResults().size() == 1);

        rewriter.setInsertionPoint(then_yield);
        mlir::Value then_func = then_yield.getResults()[0];
        mlir::Value then_pred =
            rewriter.create<qwerty::FuncPredOp>(
                func_pred_op.getLoc(), func_pred_op.getPred(),
                then_func).getResult();
        rewriter.create<mlir::scf::YieldOp>(then_yield.getLoc(),
                                            then_pred);
        rewriter.eraseOp(then_yield);

        rewriter.setInsertionPoint(else_yield);
        mlir::Value else_func = else_yield.getResults()[0];
        mlir::Value else_pred =
            rewriter.create<qwerty::FuncPredOp>(
                func_pred_op.getLoc(), func_pred_op.getPred(),
                else_func).getResult();
        rewriter.create<mlir::scf::YieldOp>(else_yield.getLoc(),
                                            else_pred);
        rewriter.eraseOp(else_yield);

        rewriter.replaceOp(func_pred_op, new_if_op.getResults());
        // Now that the old scf.if has no uses, we can finally erase it
        rewriter.eraseOp(if_op);
        return mlir::success();
    }
};


struct SimplifyPackUnpack : public mlir::OpRewritePattern<qwerty::QBundleUnpackOp> {
    using OpRewritePattern<qwerty::QBundleUnpackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleUnpackOp unpack,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value qbundle = unpack.getQbundle();
        qwerty::QBundlePackOp pack = qbundle.getDefiningOp<qwerty::QBundlePackOp>();
        if (!pack) {
            return mlir::failure();
        }
        // If the qbundle is used multiple times, it must be used in different
        // branches of a conditional. In this case, we need to do this rewrite
        // on both paths of the conditional or we violate linearity. We can't
        // guarantee that safely here, so just don't bother simplifying
        if (!pack.getQbundle().hasOneUse()) {
            return mlir::failure();
        }
        rewriter.replaceOp(unpack, pack.getQubits());
        return mlir::success();
    }
};

// Similar to SimplifyPackUnpack above
struct SimplifyBitPackUnpack : public mlir::OpRewritePattern<qwerty::BitBundleUnpackOp> {
    using OpRewritePattern<qwerty::BitBundleUnpackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::BitBundleUnpackOp unpack,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value bundle = unpack.getBundle();
        qwerty::BitBundlePackOp pack = bundle.getDefiningOp<qwerty::BitBundlePackOp>();
        if (!pack) {
            return mlir::failure();
        }
        rewriter.replaceOp(unpack, pack.getBits());
        return mlir::success();
    }
};

struct RemoveIdentity : public mlir::OpRewritePattern<qwerty::QBundleIdentityOp> {
    using OpRewritePattern<qwerty::QBundleIdentityOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::QBundleIdentityOp id,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value qbundle = id.getQbundleIn();
        rewriter.replaceOp(id, qbundle);
        return mlir::success();
    }
};

// If a user writes something like '0' or '1'@(17*pi), then replace that with
// '0' or '1'@pi. This allows our peephole optimizer to recognize Ry(theta)
// gates emitted by superpos op lowering better.
struct NormalizeSuperposTilt : public mlir::OpRewritePattern<qwerty::SuperposOp> {
    using OpRewritePattern<qwerty::SuperposOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qwerty::SuperposOp superpos_op,
                                        mlir::PatternRewriter &rewriter) const override {
        qwerty::SuperposAttr superpos = superpos_op.getSuperpos();
        llvm::SmallVector<qwerty::SuperposElemAttr> new_elems;

        bool changed = false;
        for (qwerty::SuperposElemAttr elem : superpos.getElems()) {
            double theta = elem.getPhase().getValueAsDouble();
            double norm_theta = theta;

            while (norm_theta < -M_PI - ATOL/2.0) {
                norm_theta += 2.0*M_PI;
            }
            while (M_PI + ATOL/2.0 < norm_theta) {
                norm_theta -= 2.0*M_PI;
            }

            if (std::abs(norm_theta - theta) < ATOL) {
                new_elems.push_back(elem);
            } else {
                changed = true;
                new_elems.push_back(rewriter.getAttr<qwerty::SuperposElemAttr>(
                    elem.getProb(),
                    rewriter.getF64FloatAttr(norm_theta),
                    elem.getVectors()));
            }
        }

        if (!changed) {
            return mlir::failure();
        }

        qwerty::SuperposAttr new_superpos =
            rewriter.getAttr<qwerty::SuperposAttr>(new_elems);
        rewriter.replaceOpWithNewOp<qwerty::SuperposOp>(superpos_op, new_superpos);
        return mlir::success();
    }
};
} // namespace

namespace qwerty {

//////// FUNCS /////////

// FunctionOpInterface Methods

mlir::Type FuncOp::getFunctionType() {
    return getQwertyFuncType();
}

void FuncOp::setFunctionTypeAttr(mlir::TypeAttr ty) {
    setQwertyFuncTypeAttr(ty);
}

// Need to override this because the default implementation tries to
// pass an mlir::FunctionType to setFunctionTypeAttr, which will not
// work in this case (instead we have a qwerty::FunctionType)
mlir::Type FuncOp::cloneTypeWith(mlir::TypeRange inputs,
                                    mlir::TypeRange results) {
    FunctionType qw_func_type = getQwertyFuncType();
    return FunctionType::get(qw_func_type.getContext(),
                             qw_func_type.getFunctionType().clone(inputs, results),
                             qw_func_type.getReversible());
}

// Need to override the default implementation because it requires that
// the region has as many arguments as the function type has inputs.
// This may not be the case for this operation; any additional leading
// block args are captures, actually.
mlir::LogicalResult FuncOp::verifyBody() {
    size_t n_func_args = getQwertyFuncType().getFunctionType().getInputs().size();
    // Extra arguments are captures
    if (getBody().getNumArguments() >= n_func_args) {
        return mlir::success();
    } else {
        return mlir::failure();
    }
}

// CallableOpInterfaceMethods

// mfw using pointers as optionals
::mlir::Region *FuncOp::getCallableRegion() {
    return &getBody();
}

mlir::ArrayRef<mlir::Type> FuncOp::getArgumentTypes() {
    return getQwertyFuncType().getFunctionType().getInputs();
}

mlir::ArrayRef<mlir::Type> FuncOp::getResultTypes() {
    return getQwertyFuncType().getFunctionType().getResults();
}

// Parsing/Printing

// Lifted from mlir/lib/Interfaces/FunctionImplementation.cpp
mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {

    (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

    mlir::StringAttr sym_name;
    if (parser.parseSymbolName(sym_name, mlir::SymbolTable::getSymbolAttrName(),
                               result.attributes)) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::OpAsmParser::Argument> body_args;
    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Square, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::Argument arg;
                if (parser.parseArgument(arg, /*allowType=*/true, /*allowAttrs=*/false)) {
                    return mlir::failure();
                }
                body_args.push_back(arg);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> func_arg_types;
    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::Argument arg;
                if (parser.parseArgument(arg, /*allowType=*/true, /*allowAttrs=*/false)) {
                    return mlir::failure();
                }
                body_args.push_back(arg);
                func_arg_types.push_back(arg.type);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    bool rev = false;
    if (!parser.parseOptionalKeyword("rev")) {
        rev = true;
    } else if (parser.parseKeyword("irrev")) {
        return {};
    }

    llvm::SmallVector<mlir::Type> results;
    if (!parser.parseOptionalArrow()) {
        if (parser.parseOptionalLParen()) {
            mlir::Type result;
            if (parser.parseType(result)) {
                return mlir::failure();
            }
            results.push_back(result);
        } else if (!parser.parseOptionalRParen()) {
            // We're good, just parsed ()
        } else {
            if (parser.parseCommaSeparatedList(
                    mlir::OpAsmParser::Delimiter::None, [&]() -> mlir::ParseResult {
                        mlir::Type result;
                        if (parser.parseType(result)) {
                            return mlir::failure();
                        }
                        results.push_back(result);
                        return mlir::success();
                    })) {
                return mlir::failure();
            }

            if (parser.parseRParen()) {
                return mlir::failure();
            }
        }
    }

    FunctionType qw_func_type = FunctionType::get(
        result.getContext(),
        mlir::FunctionType::get(result.getContext(), func_arg_types, results),
        rev);
    result.addAttribute(getQwertyFuncTypeAttrName(result.name),
                        mlir::TypeAttr::get(qw_func_type));

    // Why print out the `attributes' keyword before the attribute dict?
    // The keyword token resolves a parsing ambiguity where the opening { of
    // the region looks like the start of an attribute dict
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
        return mlir::failure();
    }

    // Introduce the body region and parse it.
    mlir::Region *body = result.addRegion();
    if (parser.parseRegion(*body, body_args)) {
        return mlir::failure();
    }

    return mlir::success();
}

// Lifted from mlir/lib/Interfaces/FunctionImplementation.cpp
void FuncOp::print(mlir::OpAsmPrinter &p) {
    p << ' ';
    if (getSymVisibility()) {
        p << getSymVisibility().value() << ' ';
    }
    p.printSymbolName(getSymName());

    mlir::Region &body = getBody();
    p << '[';
    size_t n_captures = getNumCaptures();
    for (size_t i = 0; i < n_captures; i++) {
        if (i > 0) {
            p << ", ";
        }
        p.printRegionArgument(body.getArgument(i));
    }
    p << "](";

    for (size_t i = n_captures; i < body.getNumArguments(); i++) {
        if (i > n_captures) {
            p << ", ";
        }
        p.printRegionArgument(body.getArgument(i));
    }
    p << ") ";

    if (getQwertyFuncType().getReversible()) {
        p << "rev";
    } else {
        p << "irrev";
    }

    mlir::TypeRange result_types =
        getQwertyFuncType().getFunctionType().getResults();
    if (!result_types.empty()) {
        p << "-> ";
        bool needsParens = result_types.size() > 1
                           || llvm::isa<mlir::FunctionType>(result_types[0]);
        if (needsParens) {
            p << '(';
        }
        for (size_t j = 0; j < result_types.size(); j++) {
            if (j) {
                p << ", ";
            }
            p.printType(result_types[j]);
        }
        if (needsParens) {
            p << ')';
        }
    }

    // Why print out the `attributes' keyword before the attribute dict?
    // The keyword token resolves a parsing ambiguity where the opening { of
    // the region looks like the start of an attribute dict
    p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), {
        getQwertyFuncTypeAttrName(), getSymNameAttrName(),
        getSymVisibilityAttrName()
    });

    p << ' ';
    p.printRegion(body,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

//////// CALLS ////////

// These print() and parse() implementations are largely based on what
// mlir-tblgen generated for a similar assemblyFormat
mlir::ParseResult CallOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
    bool adj = !parser.parseOptionalKeyword("adj");
    result.getOrAddProperties<CallOp::Properties>().adj = parser.getBuilder().getBoolAttr(adj);

    if (!parser.parseOptionalKeyword("pred")) {
        BasisAttr pred_basis;
        if (parser.parseCustomAttributeWithFallback<BasisAttr>(pred_basis)) {
            return mlir::failure();
        }
        result.getOrAddProperties<CallOp::Properties>().pred = pred_basis;
    }

    mlir::FlatSymbolRefAttr callee;
    if (parser.parseAttribute(callee, parser.getBuilder().getType<mlir::NoneType>())) {
        return mlir::failure();
    }
    result.getOrAddProperties<CallOp::Properties>().callee = callee;

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> call_operands;
    if (parser.parseOperandList(call_operands, mlir::AsmParser::Delimiter::Paren)) {
        return mlir::failure();
    }
    llvm::SMLoc arg_parse_loc = parser.getCurrentLocation();

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    if (parser.parseColon()) {
        return mlir::failure();
    }

    mlir::FunctionType all_types;
    if (parser.parseType(all_types)) {
        return mlir::failure();
    }
    result.addTypes(all_types.getResults());

    if (parser.resolveOperands(call_operands, all_types.getInputs(),
                               arg_parse_loc, result.operands)) {
        return mlir::failure();
    }

    return mlir::success();
}

void CallOp::print(mlir::OpAsmPrinter &p) {
    p << ' ';
    if (getAdj()) {
        p << "adj ";
    }
    if (getPred()) {
        p << "pred ";
        p.printStrippedAttrOrType(getPredAttr());
        p << ' ';
    }
    p.printAttributeWithoutType(getCalleeAttr());
    p << '(';
    p << getCapturesAndOperands();
    p << ')';
    p.printOptionalAttrDict(getOperation()->getAttrs(), getAttributeNames());
    p << " : ";
    p.printFunctionalType(getCapturesAndOperands().getTypes(), getOperation()->getResultTypes());
}

void CallOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
    results.add<CallWithPointlessPred>(context);
}

// CallOpInterface Methods

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
    return getCalleeAttr();
}

void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
    setCalleeAttr(llvm::cast<mlir::FlatSymbolRefAttr>(llvm::cast<mlir::SymbolRefAttr>(callee)));
}

mlir::Operation::operand_range CallOp::getArgOperands() {
    return getCapturesAndOperands();
}

mlir::MutableOperandRange CallOp::getArgOperandsMutable() {
    return getCapturesAndOperandsMutable();
}

// SymbolUserOpInterface Methods

mlir::LogicalResult CallOp::verifySymbolUses(
        mlir::SymbolTableCollection &symbolTable) {
    FuncOp func;
    if (!(func = symbolTable.lookupNearestSymbolFrom<FuncOp>(getOperation(), getCalleeAttr()))) {
        return emitOpError("no func op named ") << getCallee();
    }

    FunctionType func_type = func.getQwertyFuncType();
    if (getAdj() && !func_type.getReversible()) {
        return emitOpError("cannot call adjoint of non-reversible function");
    }

    size_t n_captures = func.getNumCaptures();
    auto callee_args = getArgOperands();
    mlir::FunctionType inner_func_type = func.getQwertyFuncType().getFunctionType();
    if (n_captures + inner_func_type.getNumInputs()
            != callee_args.size()) {
        return emitOpError("wrong number of args");
    }

    for (auto [i, func_ty, call_ty] :
            llvm::enumerate(func.getBody().getArgumentTypes(),
                            callee_args.getTypes())) {
        if (i == callee_args.size()-1 && getPred()) {
            if (!llvm::isa<qwerty::QBundleType>(func_ty)) {
                return emitOpError("last argument of reversible function "
                                   "should be a qbundle, not ") << func_ty;
            }
            qwerty::QBundleType arg_ty =
                llvm::cast<qwerty::QBundleType>(func_ty);
            func_ty = qwerty::QBundleType::get(
                arg_ty.getContext(), arg_ty.getDim() + getPredAttr().getDim());
        }

        if (func_ty != call_ty) {
            return emitOpError("argument ") << i << " has the wrong type: "
                                            << func_ty << " != " << call_ty;
        }
    }

    if (inner_func_type.getNumResults() != getNumResults()) {
        return emitOpError("wrong number of results");
    }

    if (getPred()) {
        QBundleType ret_bundle_type;
        QBundleType func_ret_bundle_type;
        size_t dim_expected = (size_t)-1;
        if (getResultTypes().size() != 1
                || !(ret_bundle_type = llvm::dyn_cast<QBundleType>(
                     getResultTypes()[0]))
                || !(func_ret_bundle_type = llvm::dyn_cast<QBundleType>(
                     inner_func_type.getResults()[0]))
                || ret_bundle_type.getDim()
                   != (dim_expected =
                       func_ret_bundle_type.getDim()
                       + getPredAttr().getDim())) {
            return emitOpError("expected a single return type of ")
                               << "!qwerty<qbundle["
                               << (dim_expected == (size_t)-1
                                   ? "?" : std::to_string(dim_expected))
                               << "]> "
                               << "for a predicated call but got "
                               << getResultTypes();
        }
    } else {
        for (auto [i, func_ty, call_ty] :
                llvm::enumerate(inner_func_type.getResults(), getResultTypes())) {
            if (func_ty != call_ty) {
                return emitOpError("result ") << i << " has the wrong type: "
                                              << func_ty << " != " << call_ty;
            }
        }
    }

    return mlir::success();
}

void CallOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    CallOpAdaptor adaptor(newInputs);
    CallOp new_call =
        rewriter.create<CallOp>(getLoc(), getResultTypes(), getCallee(),
                               !getAdj(), getPredAttr(),
                               adaptor.getCapturesAndOperands());
    newOutputs.clear();
    newOutputs.append(new_call.getResults().begin(),
                      new_call.getResults().end());
}

void CallOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    CallOpAdaptor adaptor(newInputs);
    size_t pred_dim = predBasis.getDim();

    // Take tensor product of arguments
    // We make a slightly hair-raising assumption here: there is 1 argument and
    // it is a qbundle. The rest are captures.
    mlir::ValueRange captures_and_operands = adaptor.getCapturesAndOperands();
    assert(!captures_and_operands.empty());
    size_t n_captures = captures_and_operands.size()-1;
    llvm::SmallVector<mlir::Value> captures(
        captures_and_operands.begin(),
        captures_and_operands.begin()+n_captures);
    mlir::Value arg_qbundle = captures_and_operands[n_captures];

    mlir::ValueRange unpacked_pred = rewriter.create<QBundleUnpackOp>(
        getLoc(), predIn).getQubits();
    mlir::ValueRange unpacked_args = rewriter.create<QBundleUnpackOp>(
        getLoc(), arg_qbundle).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_args.begin(), unpacked_args.end());
    mlir::Value repacked = rewriter.create<QBundlePackOp>(
        getLoc(), merged_qubits).getQbundle();
    mlir::SmallVector<mlir::Value> new_captures_and_operands(captures);
    new_captures_and_operands.push_back(repacked);

    // Adjust result type (tensor product)
    assert(getResultTypes().size() == 1);
    qwerty::QBundleType result =
        llvm::cast<qwerty::QBundleType>(getResultTypes()[0]);
    qwerty::QBundleType new_result =
        rewriter.getType<qwerty::QBundleType>(result.getDim() + pred_dim);

    // Tensor product of predicating basis
    qwerty::BasisAttr pred = getPredAttr();
    if (pred) {
        llvm::SmallVector<qwerty::BasisElemAttr> elems(
            predBasis.getElems().begin(), predBasis.getElems().end());
        elems.append(pred.getElems().begin(), pred.getElems().end());
        pred = rewriter.getAttr<qwerty::BasisAttr>(elems);
    } else {
        pred = predBasis;
    }

    CallOp new_call =
        rewriter.create<CallOp>(getLoc(), new_result, getCallee(),
                                getAdj(), pred, new_captures_and_operands);

    assert(new_call.getResults().size() == 1);
    mlir::Value qbundle_out = new_call.getResults()[0];
    mlir::ValueRange out_unpacked = rewriter.create<QBundleUnpackOp>(
        getLoc(), qbundle_out).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        out_unpacked.begin(), out_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        out_unpacked.begin() + pred_dim, out_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<QBundlePackOp>(
        getLoc(), pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<QBundlePackOp>(
        getLoc(), res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

//////// FUNCTION CONSTANT ////////

mlir::LogicalResult FuncConstOp::verifySymbolUses(
        mlir::SymbolTableCollection &symbolTable) {
    FuncOp func;
    if (!(func = symbolTable.lookupNearestSymbolFrom<FuncOp>(getOperation(), getFuncAttr()))) {
        return emitOpError("no func op named ") << getFunc();
    }

    FunctionType func_type = func.getQwertyFuncType();
    if (func_type != getResult().getType()) {
        return emitOpError("return type does not match func type");
    }

    return mlir::success();
}

//////// ADJOINT ////////

mlir::LogicalResult FuncAdjointOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        FuncAdjointOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {

    FunctionType func_type =
        llvm::cast<FunctionType>(adaptor.getCallee().getType());
    assert(func_type.getReversible()
           && "Taking adjoint of non-reversible function");
    mlir::FunctionType inner_func_type = func_type.getFunctionType();
    mlir::FunctionType reversed_inner_func_type =
        mlir::FunctionType::get(ctx, inner_func_type.getResults(),
                                inner_func_type.getInputs());
    FunctionType result_type = FunctionType::get(
        ctx, reversed_inner_func_type, func_type.getReversible());

    inferredReturnTypes.push_back(result_type);
    return mlir::success();
}

void FuncAdjointOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               mlir::MLIRContext *context) {
    results.add<AdjAdj, AdjIf>(context);
}

mlir::LogicalResult FuncAdjointOp::verify() {
    if (!getCallee().getType().getReversible()) {
        return emitOpError("Cannot take adjoint of non-reversible function");
    }
    return mlir::success();
}

//////// PREDICATOR ////////

mlir::LogicalResult FuncPredOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        FuncPredOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {

    FunctionType func_type =
        llvm::cast<FunctionType>(adaptor.getCallee().getType());
    assert(func_type.getReversible()
           && "Taking adjoint of non-reversible function");
    mlir::FunctionType inner_func_type = func_type.getFunctionType();

    size_t pred_dim = adaptor.getPred().getDim();

    assert(inner_func_type.getInputs().size() == 1
           && inner_func_type.getResults().size() == 1
           && "Expected reversible function to take and return one thing "
              "(a qbundle)");
    QBundleType qbundle_in =
        llvm::dyn_cast<QBundleType>(inner_func_type.getInputs()[0]);
    [[maybe_unused]] QBundleType qbundle_out =
        llvm::dyn_cast<QBundleType>(inner_func_type.getResults()[0]);
    assert(qbundle_in
           && qbundle_out
           && qbundle_in.getDim() == qbundle_out.getDim()
           && "Expected reversible function to take and return the same type "
              "(a qbundle)");
    size_t orig_dim = qbundle_in.getDim();
    QBundleType pred_qbundle = QBundleType::get(ctx, pred_dim + orig_dim);

    mlir::FunctionType pred_inner_func_type =
        mlir::FunctionType::get(ctx, pred_qbundle, pred_qbundle);
    FunctionType result_type = FunctionType::get(
        ctx, pred_inner_func_type, func_type.getReversible());

    inferredReturnTypes.push_back(result_type);
    return mlir::success();
}

void FuncPredOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             mlir::MLIRContext *context) {
    results.add<PredIf, PredPred, PointlessPred>(context);
}

mlir::LogicalResult FuncPredOp::verify() {
    if (!getCallee().getType().getReversible()) {
        return emitOpError("Cannot predicate non-reversible function");
    }
    return mlir::success();
}

//////// CALL INDIRECT ////////

// CallOpInterface Methods

mlir::CallInterfaceCallable CallIndirectOp::getCallableForCallee() {
    return getCallee();
}

void CallIndirectOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
    getCalleeMutable().assign(llvm::cast<mlir::Value>(callee));
}

/// Get the argument callOperands in the called function.
mlir::Operation::operand_range CallIndirectOp::getArgOperands() {
    return getCallOperands();
}

mlir::MutableOperandRange CallIndirectOp::getArgOperandsMutable() {
    return getCallOperandsMutable();
}

mlir::LogicalResult CallIndirectOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        CallIndirectOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    mlir::FunctionType inner_func_type =
        llvm::cast<qwerty::FunctionType>(
            adaptor.getCallee().getType()
        ).getFunctionType();

    inferredReturnTypes.append(inner_func_type.getResults().begin(),
                               inner_func_type.getResults().end());
    return mlir::success();
}

void CallIndirectOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                 mlir::MLIRContext *context) {
    results.add<CallIndirectConst, CallIndirectIf>(context);
}

void CallIndirectOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    CallIndirectOpAdaptor adaptor(newInputs);
    mlir::Value func_adj = rewriter.create<FuncAdjointOp>(
        getLoc(), adaptor.getCallee()).getResult();
    CallIndirectOp new_call =
        rewriter.create<CallIndirectOp>(getLoc(), getResultTypes(), func_adj,
                                       adaptor.getCallOperands());
    newOutputs.clear();
    newOutputs.append(new_call.getResults().begin(),
                      new_call.getResults().end());
}

void CallIndirectOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    CallIndirectOpAdaptor adaptor(newInputs);
    size_t pred_dim = predBasis.getDim();

    // Take tensor product of arguments
    mlir::ValueRange call_operands = adaptor.getCallOperands();
    assert(call_operands.size() == 1);
    mlir::Value arg_qbundle = call_operands[0];

    mlir::ValueRange unpacked_pred = rewriter.create<QBundleUnpackOp>(
        getLoc(), predIn).getQubits();
    mlir::ValueRange unpacked_args = rewriter.create<QBundleUnpackOp>(
        getLoc(), arg_qbundle).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_args.begin(), unpacked_args.end());
    mlir::Value repacked = rewriter.create<QBundlePackOp>(
        getLoc(), merged_qubits).getQbundle();

    mlir::Value func = rewriter.create<FuncPredOp>(
        getLoc(), predBasis, adaptor.getCallee()).getResult();
    CallIndirectOp new_call =
        rewriter.create<CallIndirectOp>(getLoc(), func, repacked);

    assert(new_call.getResults().size() == 1);
    mlir::Value qbundle_out = new_call.getResults()[0];
    mlir::ValueRange out_unpacked = rewriter.create<QBundleUnpackOp>(
        getLoc(), qbundle_out).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        out_unpacked.begin(), out_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        out_unpacked.begin() + pred_dim, out_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<QBundlePackOp>(
        getLoc(), pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<QBundlePackOp>(
        getLoc(), res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

//////// LAMBDAS ////////

mlir::Region *LambdaOp::getCallableRegion() {
    return &getRegion();
}

llvm::ArrayRef<mlir::Type> LambdaOp::getArgumentTypes() {
    return getResult().getType().getFunctionType().getInputs();
}

llvm::ArrayRef<mlir::Type> LambdaOp::getResultTypes() {
    return getResult().getType().getFunctionType().getResults();
}

// Adapted from lib/IR/FunctionImplementation.cpp in MLIR
mlir::ParseResult LambdaOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> captures;
    llvm::SmallVector<mlir::OpAsmParser::Argument> args;
    llvm::SmallVector<mlir::Type> arg_types;
    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Square, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::UnresolvedOperand capture;
                if (parser.parseOperand(capture)) {
                    return mlir::failure();
                }
                captures.push_back(capture);

                if (parser.parseKeyword("as")) {
                    return mlir::failure();
                }

                mlir::OpAsmParser::Argument arg;
                if (parser.parseArgument(arg, /*allowType=*/true, /*allowAttrs=*/false)) {
                    return mlir::failure();
                }
                args.push_back(arg);
                arg_types.push_back(arg.type);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    size_t n_captures = captures.size();
    if (parser.resolveOperands(captures, arg_types,
                               parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }

    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::Argument arg;
                if (parser.parseArgument(arg, /*allowType=*/true, /*allowAttrs=*/false)) {
                    return mlir::failure();
                }
                args.push_back(arg);
                arg_types.push_back(arg.type);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    bool rev = false;
    if (!parser.parseOptionalKeyword("rev")) {
        rev = true;
    } else if (parser.parseKeyword("irrev")) {
        return {};
    }

    llvm::SmallVector<mlir::Type> results;
    if (!parser.parseOptionalArrow()) {
        if (parser.parseOptionalLParen()) {
            mlir::Type result;
            if (parser.parseType(result)) {
                return mlir::failure();
            }
            results.push_back(result);
        } else if (!parser.parseOptionalRParen()) {
            // We're good, just parsed ()
        } else {
            if (parser.parseCommaSeparatedList(
                    mlir::OpAsmParser::Delimiter::None, [&]() -> mlir::ParseResult {
                        mlir::Type result;
                        if (parser.parseType(result)) {
                            return mlir::failure();
                        }
                        results.push_back(result);
                        return mlir::success();
                    })) {
                return mlir::failure();
            }

            if (parser.parseRParen()) {
                return mlir::failure();
            }
        }
    }

    llvm::SmallVector<mlir::Type> func_input_types(
        arg_types.begin() + n_captures, arg_types.end());
    result.types = {FunctionType::get(
                        result.getContext(),
                        mlir::FunctionType::get(
                            result.getContext(),
                            func_input_types,
                            results),
                        rev)};

    // Introduce the body region and parse it.
    mlir::Region *body = result.addRegion();
    if (parser.parseRegion(*body, args)) {
        return mlir::failure();
    }

    return mlir::success();
}

// Adapted from lib/IR/FunctionImplementation.cpp in MLIR
void LambdaOp::print(mlir::OpAsmPrinter &p) {
    unsigned n_captures = getNumCaptures();
    assert(n_captures == getCaptures().size());

    mlir::Region &region = getRegion();
    p << '[';
    for (size_t i = 0; i < n_captures; i++) {
        if (i > 0) {
            p << ", ";
        }
        p.printOperand(getCaptures()[i]);
        p << " as ";
        p.printRegionArgument(region.getArgument(i));
    }
    p << ']';

    p << '(';
    for (size_t i = n_captures; i < region.getNumArguments(); i++) {
        if (i > n_captures) {
            p << ", ";
        }
        p.printRegionArgument(region.getArgument(i));
    }
    p << ") ";

    if (getResult().getType().getReversible()) {
        p << "rev";
    } else {
        p << "irrev";
    }

    mlir::TypeRange result_types = getResult().getType().getFunctionType().getResults();
    if (!result_types.empty()) {
        p << "-> ";
        bool needsParens = result_types.size() > 1
                           || llvm::isa<mlir::FunctionType>(result_types[0]);
        if (needsParens) {
            p << '(';
        }
        for (size_t j = 0; j < result_types.size(); j++) {
            if (j) {
                p << ", ";
            }
            p.printType(result_types[j]);
        }
        if (needsParens) {
            p << ')';
        }
    }

    p << ' ';
    p.printRegion(getRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

// Yanked from scf.execute_region
mlir::ParseResult PredOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    if (parser.parseKeyword("on")) {
        return mlir::failure();
    }
    qwerty::BasisAttr basis;
    if (parser.parseCustomAttributeWithFallback<BasisAttr>(basis)) {
        return mlir::failure();
    }
    result.getOrAddProperties<PredOp::Properties>().basis = basis;

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::Type> operand_types;

    if (parser.parseLParen()) {
        return mlir::failure();
    }
    // First, parse pred bundle
    mlir::OpAsmParser::UnresolvedOperand operand;
    mlir::Type operand_type;
    if (parser.parseOperand(operand)
            || parser.parseColon()
            || parser.parseType(operand_type)) {
        return mlir::failure();
    }
    operands.push_back(operand);
    operand_types.push_back(operand_type);

    if (parser.parseComma()) {
        return mlir::failure();
    }

    mlir::OpAsmParser::Argument arg;
    if (parser.parseOperand(operand)
            || parser.parseKeyword("as")
            || parser.parseArgument(arg, /*allowType=*/true)) {
        return mlir::failure();
    }
    operands.push_back(operand);
    operand_types.push_back(arg.type);

    if (parser.parseRParen()) {
        return mlir::failure();
    }

    if (parser.resolveOperands(
                operands, operand_types, parser.getCurrentLocation(),
                result.operands)
            || parser.parseArrowTypeList(result.types)) {
        return mlir::failure();
    }

    // Introduce the body region and parse it.
    mlir::Region *body = result.addRegion();
    if (parser.parseRegion(*body, arg) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    return mlir::success();
}

// Yanked from scf.execute_region
void PredOp::print(mlir::OpAsmPrinter &p) {
    p << " on ";
    p.printStrippedAttrOrType(getBasis());
    p << " (";
    p.printOperand(getPredBundleIn());
    p << ": ";
    p.printType(getPredBundleIn().getType());
    p << ", ";
    p.printOperand(getRegionArg());
    p << " as ";
    p.printRegionArgument(getRegion().front().getArgument(0));
    p << ')';
    p.printArrowTypeList(getResultTypes());

    p << ' ';
    p.printRegion(getRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);

    p.printOptionalAttrDict((*this)->getAttrs(), PredOp::getAttributeNames());
}

// Based on scf.execute_region
void PredOp::getSuccessorRegions(
        mlir::RegionBranchPoint point, llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
    // Branch from parent region into this region
    if (point.isParent()) {
        regions.push_back(mlir::RegionSuccessor(&getRegion()));
        return;
    }
    // Branch back into parent region
    regions.push_back(mlir::RegionSuccessor(getResults()));
}

mlir::LogicalResult PredOp::verify() {
    auto pred_bundle = getPredBundleOut();
    if (!(pred_bundle.hasOneUse() || linearCheckForManyUses(pred_bundle))) {
        return emitOpError("Predicate qubits are not linear with this IR "
                           "instruction");
    }

    auto res_bundle = getRegionResult();
    if (!(res_bundle.hasOneUse() || linearCheckForManyUses(res_bundle))) {
        return emitOpError("Result qubits are not linear with this IR "
                           "instruction");
    }

    if (!getRegion().hasOneBlock()) {
        return emitOpError("Must have 1 block");
    }
    mlir::Block &block = getRegion().front();
    if (block.getNumArguments() != 1
            || !llvm::isa<QBundleType>(block.getArgument(0).getType())) {
        return emitOpError("Block should have 1 qbundle argument but instead "
                           "has the following argument types: ")
                          << block.getArgumentTypes();
    }

    if (getRegionArg().getType() != block.getArgument(0).getType()) {
        return emitOpError("Block argument type should be the same as the "
                           "regionArg operand. Yet ")
                          << getRegionArg().getType() << " != "
                          << block.getArgument(0).getType();
    }

    return mlir::success();
}

void PredOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    PredOpAdaptor adaptor(newInputs);
    assert(!getBasis().hasPhases());

    mlir::Region &region = getRegion();
    assert(region.hasOneBlock());
    mlir::Block &block = region.front();

    mlir::Location loc = getLoc();
    [[maybe_unused]] auto res =
        qcirc::takeAdjointOfBlockInPlace<qwerty::YieldOp>(
            rewriter, block, loc);
    assert(mlir::succeeded(res) && "taking adjoint of PredOp failed");

    PredOp new_pred = rewriter.create<PredOp>(
        loc, getPredBundleOut().getType(), getRegionResult().getType(),
        getBasis(), adaptor.getPredBundleIn(), adaptor.getRegionArg());

    mlir::Region &new_region = new_pred.getRegion();
    // TODO: Is it risky to more or less gut this op? (This moves the block in
    //       the region and leaves the old region empty.) In the current
    //       implementation, it is fine.
    rewriter.inlineRegionBefore(region, new_region, new_region.end());

    newOutputs.clear();
    newOutputs.push_back(new_pred.getPredBundleOut());
    newOutputs.push_back(new_pred.getRegionResult());
}

void PredOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    assert(!getBasis().hasPhases());
    PredOpAdaptor adaptor(newInputs);
    mlir::Location loc = getLoc();

    mlir::Region &region = getRegion();
    assert(region.hasOneBlock());

    mlir::ValueRange new_pred_qubits =
        rewriter.create<QBundleUnpackOp>(loc, predIn).getQubits();
    mlir::ValueRange old_pred_qubits =
        rewriter.create<QBundleUnpackOp>(
            loc, adaptor.getPredBundleIn()).getQubits();
    llvm::SmallVector<mlir::Value> pred_qubits(new_pred_qubits);
    pred_qubits.append(old_pred_qubits.begin(), old_pred_qubits.end());
    mlir::Value pred_qbundle =
        rewriter.create<QBundlePackOp>(loc, pred_qubits).getQbundle();

    llvm::SmallVector<BasisElemAttr> elems(predBasis.getElems());
    elems.append(getBasis().getElems().begin(), getBasis().getElems().end());
    BasisAttr basis = rewriter.getAttr<BasisAttr>(elems);

    PredOp new_pred = rewriter.create<PredOp>(
        loc, pred_qbundle.getType(), getRegionResult().getType(),
        basis, pred_qbundle, adaptor.getRegionArg());

    mlir::Region &new_region = new_pred.getRegion();
    // TODO: Is it risky to more or less gut this op? (This moves the block in
    //       the region and leaves the old region empty.) In the current
    //       implementation, it is fine.
    rewriter.inlineRegionBefore(region, new_region, new_region.end());

    mlir::ValueRange post_pred_qubits =
        rewriter.create<QBundleUnpackOp>(
            loc, new_pred.getPredBundleOut()).getQubits();
    llvm::SmallVector<mlir::Value> new_post_pred_qubits(
        post_pred_qubits.begin(),
        post_pred_qubits.begin() + predBasis.getDim());
    llvm::SmallVector<mlir::Value> old_post_pred_qubits(
        post_pred_qubits.begin() + predBasis.getDim(),
        post_pred_qubits.end());
    mlir::Value new_post_pred_qbundle =
        rewriter.create<QBundlePackOp>(loc, new_post_pred_qubits).getQbundle();
    mlir::Value old_post_pred_qbundle =
        rewriter.create<QBundlePackOp>(loc, old_post_pred_qubits).getQbundle();

    predOut = new_post_pred_qbundle;
    newOutputs.clear();
    newOutputs.push_back(old_post_pred_qbundle);
    newOutputs.push_back(new_pred.getRegionResult());
}

//////// QBUNDLE MANAGEMENT ////////

void QBundleUnpackOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                  mlir::MLIRContext *context) {
    results.add<SimplifyPackUnpack>(context);
}

void QBundleIdentityOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                    mlir::MLIRContext *context) {
    results.add<RemoveIdentity>(context);
}

mlir::LogicalResult QBundlePrepOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundlePrepOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    size_t n_qubits = adaptor.getDim();
    QBundleType retType = QBundleType::get(ctx, n_qubits);
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

mlir::LogicalResult QBundlePrepOp::verify() {
    auto result = getResult();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
        return this->emitOpError("QBundlePrepOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

bool QBundlePrepOp::isAdjointable() {
    return getPrimBasis() == PrimitiveBasis::Z && getEigenstate() == Eigenstate::PLUS;
}

unsigned QBundlePrepOp::getNumOperandsOfAdjoint() {
    return 1; // Will become a discardz
}

void QBundlePrepOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(isAdjointable());
    QBundleDiscardZeroOpAdaptor adaptor(newInputs);
    rewriter.create<QBundleDiscardZeroOp>(getLoc(), adaptor.getQbundle());
    newOutputs.clear();
}

mlir::LogicalResult QBundlePackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundlePackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    size_t n_qubits = adaptor.getQubits().size();
    QBundleType retType = QBundleType::get(ctx, n_qubits);
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

mlir::LogicalResult QBundlePackOp::verify() {
    auto bundle = getQbundle();

    if (!(bundle.hasOneUse() || linearCheckForManyUses(bundle))) {
        return this->emitOpError("QBundlePackOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

unsigned QBundlePackOp::getNumOperandsOfAdjoint() {
    return 1; // Will become a qbunpack
}

void QBundlePackOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleUnpackOpAdaptor adaptor(newInputs);
    QBundleUnpackOp unpack = rewriter.create<QBundleUnpackOp>(getLoc(), adaptor.getQbundle());
    newOutputs.clear();
    newOutputs.append(unpack.getQubits().begin(), unpack.getQubits().end());
}

mlir::LogicalResult QBundleUnpackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleUnpackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType bundle = llvm::dyn_cast<QBundleType>(adaptor.getQbundle().getType());
    if (!bundle) {
        return mlir::failure();
    }
    size_t n_qubits = bundle.getDim();
    inferredReturnTypes.append(n_qubits, qcirc::QubitType::get(ctx));
    return mlir::success();
}

mlir::LogicalResult QBundleUnpackOp::verify() {
    auto qubits = getQubits();

    for (auto indexedResult : llvm::enumerate(qubits)) {
        mlir::Value qubit = indexedResult.value();

        if (!(qubit.hasOneUse() || linearCheckForManyUses(qubit))) {
            return this->emitOpError("QBundleUnpackOp: ")
                << "Qubit(" << indexedResult.index()
                << ") is not linear with this IR instruction (gate)";
        }
    }

    return mlir::success();
}

unsigned QBundleUnpackOp::getNumOperandsOfAdjoint() {
    return getQubits().size(); // Will become a qbpack
}

void QBundleUnpackOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundlePackOpAdaptor adaptor(newInputs);
    QBundlePackOp pack = rewriter.create<QBundlePackOp>(getLoc(), adaptor.getQubits());
    newOutputs.clear();
    newOutputs.push_back(pack.getQbundle());
}

//////// INITIALIZATION ////////

mlir::LogicalResult BitInitOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        BitInitOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    qwerty::BitBundleType bitBundleType = llvm::dyn_cast<qwerty::BitBundleType>(adaptor.getBitBundle().getType());
    if (!bitBundleType) {
        return mlir::failure();
    }
    QBundleType retType = QBundleType::get(ctx, bitBundleType.getDim());
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

mlir::LogicalResult BitInitOp::verify() {
    if (getQbundleIn().getType().getDim() != getQbundleOut().getType().getDim()
            || getQbundleIn().getType().getDim() != getBitBundle().getType().getDim()) {
        return this->emitOpError("Dimensions misaligned for BitInitOp") ;
    }
    return mlir::success();
}

void BitInitOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    BitInitOpAdaptor adaptor(newInputs);
    BitInitOp init = rewriter.create<BitInitOp>(
        getLoc(), adaptor.getBitBundle(), adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(init.getQbundleOut());
}

mlir::LogicalResult QBundleInitOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleInitOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    inferredReturnTypes.insert(inferredReturnTypes.end(),
                               adaptor.getQbundleIn().getType());
    return mlir::success();
}

mlir::LogicalResult QBundleInitOp::verify() {
    if (getQbundleIn().getType().getDim() != getQbundleOut().getType().getDim()
            || getQbundleIn().getType().getDim() != getBasis().getDim()) {
        return this->emitOpError("Dimensions misaligned for QBundleInitOp") ;
    }

    if (getBasis().getNumPhases() != getBasisPhases().size()) {
        return emitOpError("Mismatch in number of basis phases");
    }

    return mlir::success();
}

void QBundleInitOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleDeinitOpAdaptor adaptor(newInputs);
    QBundleDeinitOp deinit = rewriter.create<QBundleDeinitOp>(getLoc(),
        getBasis(), adaptor.getBasisPhases(), adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(deinit.getQbundleOut());
}

void QBundleInitOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleInitOpAdaptor adaptor(newInputs);
    buildPredicatedInit(getLoc(), getBasis(), adaptor.getBasisPhases(),
                        adaptor.getQbundleIn(), /*reverse=*/false, rewriter,
                        predBasis, predIn, predOut, newOutputs);
}

mlir::LogicalResult QBundleDeinitOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleDeinitOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    inferredReturnTypes.insert(inferredReturnTypes.end(),
                               adaptor.getQbundleIn().getType());
    return mlir::success();
}

mlir::LogicalResult QBundleDeinitOp::verify() {
    if (getQbundleIn().getType().getDim() != getQbundleOut().getType().getDim()
            || getQbundleIn().getType().getDim() != getBasis().getDim()) {
        return this->emitOpError("Dimensions misaligned for QBundleDeinitOp") ;
    }

    if (getBasis().getNumPhases() != getBasisPhases().size()) {
        return emitOpError("Mismatch in number of basis phases");
    }

    return mlir::success();
}

void QBundleDeinitOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleInitOpAdaptor adaptor(newInputs);
    QBundleInitOp init = rewriter.create<QBundleInitOp>(getLoc(),
        getBasis(), adaptor.getBasisPhases(), adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(init.getQbundleOut());
}

void QBundleDeinitOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleDeinitOpAdaptor adaptor(newInputs);
    buildPredicatedInit(getLoc(), getBasis(), adaptor.getBasisPhases(),
                        adaptor.getQbundleIn(), /*reverse=*/true, rewriter,
                        predBasis, predIn, predOut, newOutputs);
}

//////// QBUNDLE OPERATIONS ////////

unsigned QBundleDiscardZeroOp::getNumOperandsOfAdjoint() {
    return 0; // Will become a qbprep
}

void QBundleDiscardZeroOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(newInputs.empty());
    mlir::Value qbundle = rewriter.create<QBundlePrepOp>(
            getLoc(), PrimitiveBasis::Z, Eigenstate::PLUS, getQbundle().getType().getDim()
        ).getResult();

    newOutputs.clear();
    newOutputs.push_back(qbundle);
}

mlir::LogicalResult QBundleIdentityOp::verify() {
    auto bundle = getQbundleOut();

    if (!(bundle.hasOneUse() || linearCheckForManyUses(bundle))) {
        return this->emitOpError("QBundleIdentityOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

void QBundleIdentityOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleIdentityOpAdaptor adaptor(newInputs);
    QBundleIdentityOp id = rewriter.create<QBundleIdentityOp>(getLoc(), adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(id.getQbundleOut());
}

mlir::LogicalResult QBundlePhaseOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundlePhaseOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType qbundleType = llvm::dyn_cast<QBundleType>(adaptor.getQbundleIn().getType());
    if (!qbundleType) {
        return mlir::failure();
    }
    inferredReturnTypes.insert(inferredReturnTypes.end(), qbundleType);
    return mlir::success();
}

mlir::LogicalResult QBundlePhaseOp::verify() {
    auto bundle = getQbundleOut();

    if (!(bundle.hasOneUse() || linearCheckForManyUses(bundle))) {
        return this->emitOpError("QBundlePhaseOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

void QBundlePhaseOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundlePhaseOpAdaptor adaptor(newInputs);
    mlir::Value neg_theta = qcirc::stationaryF64Negate(
        rewriter, getLoc(), adaptor.getTheta());
    QBundlePhaseOp phase = rewriter.create<QBundlePhaseOp>(
        getLoc(), neg_theta, adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(phase.getQbundleOut());
}

void QBundlePhaseOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    QBundlePhaseOpAdaptor adaptor(newInputs);
    mlir::Location loc = getLoc();
    size_t pred_dim = predBasis.getDim();
    size_t dim = getQbundleIn().getType().getDim();

    mlir::ValueRange unpacked_pred = rewriter.create<QBundleUnpackOp>(
        loc, predIn).getQubits();
    mlir::ValueRange unpacked_in = rewriter.create<QBundleUnpackOp>(
        loc, adaptor.getQbundleIn()).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_in.begin(), unpacked_in.end());
    mlir::Value repacked = rewriter.create<QBundlePackOp>(
        loc, merged_qubits).getQbundle();

    // Convert pred & @theta to the following
    // basis translation:
    // pred + std[N] >> pred + std[N-1] + {'0'@theta, '1'@theta}
    llvm::SmallVector<BasisElemAttr> lhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    lhs_elems.push_back(
        rewriter.getAttr<BasisElemAttr>(
            rewriter.getAttr<BuiltinBasisAttr>(
                PrimitiveBasis::Z, dim)));

    llvm::SmallVector<BasisElemAttr> rhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    if (dim > 1) {
        rhs_elems.push_back(
            rewriter.getAttr<BasisElemAttr>(
                rewriter.getAttr<BuiltinBasisAttr>(
                    PrimitiveBasis::Z, dim-1)));
    }
    rhs_elems.push_back(
        rewriter.getAttr<BasisElemAttr>(
            rewriter.getAttr<BasisVectorListAttr>(
                std::initializer_list<BasisVectorAttr>{
                    rewriter.getAttr<BasisVectorAttr>(
                        PrimitiveBasis::Z, Eigenstate::PLUS,
                        /*dim=*/1, /*hasPhase=*/true),
                    rewriter.getAttr<BasisVectorAttr>(
                        PrimitiveBasis::Z, Eigenstate::MINUS,
                        /*dim=*/1, /*hasPhase=*/true)})));

    mlir::Value res = rewriter.create<QBundleBasisTranslationOp>(loc,
        rewriter.getAttr<BasisAttr>(lhs_elems),
        rewriter.getAttr<BasisAttr>(rhs_elems),
        std::initializer_list<mlir::Value>{
            adaptor.getTheta(), adaptor.getTheta()},
        repacked).getQbundleOut();

    mlir::ValueRange res_unpacked = rewriter.create<QBundleUnpackOp>(
        loc, res).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        res_unpacked.begin(), res_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        res_unpacked.begin() + pred_dim, res_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<QBundlePackOp>(
        loc, pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<QBundlePackOp>(
        loc, res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

mlir::LogicalResult QBundleBasisTranslationOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleBasisTranslationOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType qbundleType = llvm::dyn_cast<QBundleType>(adaptor.getQbundleIn().getType());
    if (!qbundleType) {
        return mlir::failure();
    }
    inferredReturnTypes.insert(inferredReturnTypes.end(), qbundleType);
    return mlir::success();
}

mlir::LogicalResult QBundleBasisTranslationOp::verify() {
    uint64_t basis_in_dim = getBasisIn().getDim();
    uint64_t qbundle_dim = getQbundleIn().getType().getDim();
    uint64_t basis_out_dim = getBasisOut().getDim();
    auto bundle = getQbundleOut();

    if (basis_in_dim != qbundle_dim) {
        return this->emitOpError("QBundleBasisTranslationOp: ")
            << "Dimension mismatch between input basis and qbundle: "
            << basis_in_dim << " != " << qbundle_dim;
    }

    if (basis_out_dim != qbundle_dim) {
        return this->emitOpError("QBundleBasisTranslationOp: ")
            << "Dimension mismatch between output basis and qbundle: "
            << basis_out_dim << " != " << qbundle_dim;
    }

    uint64_t n_total_phases =
        getBasisIn().getNumPhases() + getBasisOut().getNumPhases();
    if (n_total_phases != getBasisPhases().size()) {
        return emitOpError("Mismatch in number of basis phases");
    }

    if (!(bundle.hasOneUse() || linearCheckForManyUses(bundle))) {
        return this->emitOpError("QBundleBasisTranslationOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

void QBundleBasisTranslationOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleBasisTranslationOpAdaptor adaptor(newInputs);
    assert(getBasisIn().getNumPhases() + getBasisOut().getNumPhases()
           == adaptor.getBasisPhases().size());
    // Swap order of phases. The ordering is currently phases for the left
    // basis first and then the right basis, but we need the right basis phases
    // and then the left basis phases.
    llvm::SmallVector<mlir::Value> basis_phases(
        adaptor.getBasisPhases().begin() + getBasisIn().getNumPhases(),
        adaptor.getBasisPhases().end());
    basis_phases.append(
        adaptor.getBasisPhases().begin(),
        adaptor.getBasisPhases().begin() + getBasisIn().getNumPhases());
    QBundleBasisTranslationOp flipped =
        rewriter.create<QBundleBasisTranslationOp>(getLoc(),
            getBasisOut(), getBasisIn(), basis_phases, adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(flipped.getQbundleOut());
}

void QBundleBasisTranslationOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    QBundleBasisTranslationOpAdaptor adaptor(newInputs);
    mlir::Location loc = getLoc();
    size_t pred_dim = predBasis.getDim();

    mlir::ValueRange unpacked_pred = rewriter.create<QBundleUnpackOp>(
        loc, predIn).getQubits();
    mlir::ValueRange unpacked_in = rewriter.create<QBundleUnpackOp>(
        loc, adaptor.getQbundleIn()).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_in.begin(), unpacked_in.end());
    mlir::Value repacked = rewriter.create<QBundlePackOp>(
        loc, merged_qubits).getQbundle();

    // Convert pred & (b1 >> b2) into the following basis translation:
    // pred + b1 >> pred + b2
    llvm::SmallVector<BasisElemAttr> lhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    lhs_elems.append(getBasisIn().getElems().begin(),
                     getBasisIn().getElems().end());

    llvm::SmallVector<BasisElemAttr> rhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    rhs_elems.append(getBasisOut().getElems().begin(),
                     getBasisOut().getElems().end());

    mlir::Value res = rewriter.create<QBundleBasisTranslationOp>(loc,
        rewriter.getAttr<BasisAttr>(lhs_elems),
        rewriter.getAttr<BasisAttr>(rhs_elems),
        adaptor.getBasisPhases(),
        repacked).getQbundleOut();

    mlir::ValueRange res_unpacked = rewriter.create<QBundleUnpackOp>(
        loc, res).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        res_unpacked.begin(), res_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        res_unpacked.begin() + pred_dim, res_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<QBundlePackOp>(
        loc, pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<QBundlePackOp>(
        loc, res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

mlir::LogicalResult QBundleMeasureOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleMeasureOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType qbundleType = llvm::dyn_cast<QBundleType>(adaptor.getQbundle().getType());
    if (!qbundleType) {
        return mlir::failure();
    }
    qwerty::BitBundleType retType = qwerty::BitBundleType::get(ctx, qbundleType.getDim());
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

mlir::LogicalResult QBundleMeasureOp::verify() {
    uint64_t basis_dim = getBasis().getDim();
    uint64_t qbundle_dim = getQbundle().getType().getDim();

    if (basis_dim != qbundle_dim) {
        return this->emitOpError("Dimension mismatch between measurement ")
            << "basis and qbundle: " << basis_dim << " != " << qbundle_dim;
    }

    return mlir::success();
}

mlir::LogicalResult QBundleProjectOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleProjectOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType qbundleInType = llvm::dyn_cast<QBundleType>(adaptor.getQbundleIn().getType());
    if (!qbundleInType) {
        return mlir::failure();
    }
    QBundleType retType = QBundleType::get(ctx, qbundleInType.getDim());
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

mlir::LogicalResult QBundleProjectOp::verify() {
    uint64_t basis_dim = getBasis().getDim();
    uint64_t qbundle_in_dim = getQbundleIn().getType().getDim();

    if (basis_dim != qbundle_in_dim) {
        return this->emitOpError("Dimension mismatch between measurement ")
            << "basis and input qbundle: " << basis_dim << " != " << qbundle_in_dim;
    }

    return mlir::success();
}

mlir::LogicalResult QBundleFlipOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleFlipOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType qbundleType = llvm::dyn_cast<QBundleType>(adaptor.getQbundleIn().getType());
    if (!qbundleType) {
        return mlir::failure();
    }
    inferredReturnTypes.insert(inferredReturnTypes.end(), qbundleType);
    return mlir::success();
}

mlir::LogicalResult QBundleFlipOp::verify() {
    uint64_t basis_dim = getBasis().getDim();
    uint64_t qbundle_dim = getQbundleIn().getType().getDim();

    if (basis_dim != qbundle_dim) {
        return emitOpError("QBundleFlipOp: ")
            << "Dimension mismatch between basis and qbundle: "
            << basis_dim << " != " << qbundle_dim;
    }

    if (basis_dim != 1) {
        return emitOpError("Basis for flip must be one qubit");
    }

    if (getBasis().hasPredicate()) {
        return emitOpError("Flip basis must fully span");
    }

    if (getBasis().getNumPhases() != getBasisPhases().size()) {
        return emitOpError("Mismatch in number of basis phases");
    }

    auto bundle_out = getQbundleOut();
    if (!(bundle_out.hasOneUse() || linearCheckForManyUses(bundle_out))) {
        return emitOpError("QBundleFlipOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

void QBundleFlipOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleFlipOpAdaptor adaptor(newInputs);
    // Recreate ourself again, since ~b.flip is equivalent to b.flip
    QBundleFlipOp flip =
        rewriter.create<QBundleFlipOp>(getLoc(),
            getBasis(), adaptor.getBasisPhases(), adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(flip.getQbundleOut());
}

void QBundleFlipOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    QBundleFlipOpAdaptor adaptor(newInputs);
    mlir::Location loc = getLoc();
    size_t pred_dim = predBasis.getDim();

    mlir::ValueRange unpacked_pred = rewriter.create<QBundleUnpackOp>(
        loc, predIn).getQubits();
    mlir::ValueRange unpacked_in = rewriter.create<QBundleUnpackOp>(
        loc, adaptor.getQbundleIn()).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_in.begin(), unpacked_in.end());
    mlir::Value repacked = rewriter.create<QBundlePackOp>(
        loc, merged_qubits).getQbundle();

    // Convert pred & {v1,v2}.flip into the following basis translation:
    // pred + {v1,v2} >> pred + {v2,v1}
    llvm::SmallVector<BasisElemAttr> lhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    lhs_elems.append(getBasis().getElems().begin(),
                     getBasis().getElems().end());

    llvm::SmallVector<BasisElemAttr> rhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    assert(getBasis().getDim() == 1 && !getBasis().hasPredicate());
    BasisElemAttr sole_elem = getBasis().getElems()[0];
    BasisVectorListAttr sole_veclist = sole_elem.getVeclist();
    if (!sole_veclist) {
        assert(sole_elem.getStd());
        sole_veclist = sole_elem.getStd().expandToVeclist();
    }
    rhs_elems.push_back(rewriter.getAttr<BasisElemAttr>(
        rewriter.getAttr<BasisVectorListAttr>(
            std::initializer_list<BasisVectorAttr>{
                sole_veclist.getVectors()[1], sole_veclist.getVectors()[0]})));

    llvm::SmallVector<mlir::Value> phases(adaptor.getBasisPhases());
    auto rev_phases = llvm::reverse(adaptor.getBasisPhases());
    phases.append(rev_phases.begin(), rev_phases.end());

    mlir::Value res = rewriter.create<QBundleBasisTranslationOp>(loc,
        rewriter.getAttr<BasisAttr>(lhs_elems),
        rewriter.getAttr<BasisAttr>(rhs_elems),
        phases,
        repacked).getQbundleOut();

    mlir::ValueRange res_unpacked = rewriter.create<QBundleUnpackOp>(
        loc, res).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        res_unpacked.begin(), res_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        res_unpacked.begin() + pred_dim, res_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<QBundlePackOp>(
        loc, pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<QBundlePackOp>(
        loc, res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

mlir::LogicalResult QBundleRotateOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        QBundleRotateOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType qbundleType = llvm::dyn_cast<QBundleType>(adaptor.getQbundleIn().getType());
    if (!qbundleType) {
        return mlir::failure();
    }
    inferredReturnTypes.insert(inferredReturnTypes.end(), qbundleType);
    return mlir::success();
}

mlir::LogicalResult QBundleRotateOp::verify() {
    uint64_t basis_dim = getBasis().getDim();
    uint64_t qbundle_dim = getQbundleIn().getType().getDim();

    if (basis_dim != qbundle_dim) {
        return this->emitOpError("QBundleRotateOp: ")
            << "Dimension mismatch between basis and qbundle: "
            << basis_dim << " != " << qbundle_dim;
    }

    if (basis_dim != 1) {
        return this->emitOpError("Basis for rotate must be one qubit");
    }

    if (getBasis().hasPredicate()) {
        return this->emitOpError("Rotate basis must fully span");
    }

    auto bundle_out = getQbundleOut();
    if (!(bundle_out.hasOneUse() || linearCheckForManyUses(bundle_out))) {
        return this->emitOpError("QBundleRotateOp: ")
            << "Bundle qubits is not linear with this IR instruction";
    }

    return mlir::success();
}

mlir::LogicalResult SuperposOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        SuperposOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    QBundleType retType = QBundleType::get(ctx, adaptor.getSuperpos().getDim());
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

void SuperposOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             mlir::MLIRContext *context) {
    results.add<NormalizeSuperposTilt>(context);
}

void QBundleRotateOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QBundleRotateOpAdaptor adaptor(newInputs);
    mlir::Value neg_theta = qcirc::stationaryF64Negate(
        rewriter, getLoc(), adaptor.getTheta());
    QBundleRotateOp rot =
        rewriter.create<QBundleRotateOp>(
            getLoc(), getBasis(), neg_theta, adaptor.getQbundleIn());
    newOutputs.clear();
    newOutputs.push_back(rot.getQbundleOut());
}

void QBundleRotateOp::buildPredicated(
        mlir::RewriterBase &rewriter,
        qwerty::BasisAttr predBasis,
        mlir::Value predIn,
        mlir::Value &predOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(!predBasis.hasPhases());
    assert(!getBasis().hasPhases());
    QBundleRotateOpAdaptor adaptor(newInputs);
    mlir::Location loc = getLoc();
    size_t pred_dim = predBasis.getDim();

    mlir::ValueRange unpacked_pred = rewriter.create<QBundleUnpackOp>(
        loc, predIn).getQubits();
    mlir::ValueRange unpacked_in = rewriter.create<QBundleUnpackOp>(
        loc, adaptor.getQbundleIn()).getQubits();
    llvm::SmallVector<mlir::Value> merged_qubits(unpacked_pred.begin(),
                                                 unpacked_pred.end());
    merged_qubits.append(unpacked_in.begin(), unpacked_in.end());
    mlir::Value repacked = rewriter.create<QBundlePackOp>(
        loc, merged_qubits).getQbundle();

    // Convert pred & {v1,v2}.rotate(theta) into the following basis translation:
    // pred + {v1,v2} >> pred + {v2@(-theta/2),v1@(theta/2)}
    llvm::SmallVector<BasisElemAttr> lhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    lhs_elems.append(getBasis().getElems().begin(),
                     getBasis().getElems().end());

    llvm::SmallVector<BasisElemAttr> rhs_elems(
        predBasis.getElems().begin(),
        predBasis.getElems().end());
    assert(getBasis().getDim() == 1 && !getBasis().hasPredicate());
    BasisElemAttr sole_elem = getBasis().getElems()[0];
    BasisVectorListAttr sole_veclist = sole_elem.getVeclist();
    if (!sole_veclist) {
        assert(sole_elem.getStd());
        sole_veclist = sole_elem.getStd().expandToVeclist();
    }
    BasisVectorAttr v1 = sole_veclist.getVectors()[0];
    BasisVectorAttr v2 = sole_veclist.getVectors()[1];
    rhs_elems.push_back(rewriter.getAttr<BasisElemAttr>(
        rewriter.getAttr<BasisVectorListAttr>(
            std::initializer_list<BasisVectorAttr>{
                rewriter.getAttr<BasisVectorAttr>(
                    v1.getPrimBasis(), v1.getEigenbits(), v1.getDim(), /*hasPhase=*/true),
                rewriter.getAttr<BasisVectorAttr>(
                    v2.getPrimBasis(), v2.getEigenbits(), v2.getDim(), /*hasPhase=*/true)})));

    mlir::Value theta_by_2 = wrapStationaryFloatOps(
        rewriter, loc, adaptor.getTheta(),
        [&](mlir::Value theta_arg) {
            mlir::Value const_2 = rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getF64FloatAttr(2.0)).getResult();
            return rewriter.create<mlir::arith::DivFOp>(
                loc, theta_arg, const_2).getResult();
        });
    mlir::Value neg_theta_by_2 = wrapStationaryFloatOps(
        rewriter, loc, theta_by_2,
        [&](mlir::Value theta_by_2_arg) {
            return rewriter.create<mlir::arith::NegFOp>(
                loc, theta_by_2_arg).getResult();
        });

    mlir::Value res = rewriter.create<QBundleBasisTranslationOp>(loc,
        rewriter.getAttr<BasisAttr>(lhs_elems),
        rewriter.getAttr<BasisAttr>(rhs_elems),
        std::initializer_list<mlir::Value>{neg_theta_by_2, theta_by_2},
        repacked).getQbundleOut();

    mlir::ValueRange res_unpacked = rewriter.create<QBundleUnpackOp>(
        loc, res).getQubits();
    llvm::SmallVector<mlir::Value> pred_bundle(
        res_unpacked.begin(), res_unpacked.begin() + pred_dim);
    llvm::SmallVector<mlir::Value> res_bundle(
        res_unpacked.begin() + pred_dim, res_unpacked.end());
    mlir::Value pred_repacked = rewriter.create<QBundlePackOp>(
        loc, pred_bundle).getQbundle();
    mlir::Value res_repacked = rewriter.create<QBundlePackOp>(
        loc, res_bundle).getQbundle();

    predOut = pred_repacked;
    newOutputs.clear();
    newOutputs.push_back(res_repacked);
}

mlir::LogicalResult BitBundlePackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        BitBundlePackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    size_t n_bits = adaptor.getBits().size();
    BitBundleType retType = BitBundleType::get(ctx, n_bits);
    inferredReturnTypes.insert(inferredReturnTypes.end(), retType);
    return mlir::success();
}

mlir::LogicalResult BitBundlePackOp::verify() {
    if (getBits().empty()) {
        return emitOpError("Empty bit bundle is illegal");;
    }
    return mlir::success();
}

mlir::LogicalResult BitBundleUnpackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        BitBundleUnpackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    size_t n_bits = llvm::cast<BitBundleType>(adaptor.getBundle().getType()).getDim();
    mlir::Type bit_type = mlir::IntegerType::get(ctx, 1);
    inferredReturnTypes.insert(inferredReturnTypes.end(), n_bits, bit_type);
    return mlir::success();
}

void BitBundleUnpackOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                    mlir::MLIRContext *context) {
    results.add<SimplifyBitPackUnpack>(context);
}

mlir::LogicalResult EmbedXorOp::verifySymbolUses(
        mlir::SymbolTableCollection &symbolTable) {
    ccirc::CircuitOp circ;
    if (!(circ = symbolTable.lookupNearestSymbolFrom<ccirc::CircuitOp>(getOperation(), getCircuitAttr()))) {
        return emitOpError("no circuit op named ") << getCircuit();
    }

    FunctionType func_type = getQwertyFuncTypeOf(circ);
    if (func_type != getResult().getType()) {
        return emitOpError("return type does not match func type");
    }

    return mlir::success();
}

} // namespace qwerty
