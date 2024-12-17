#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Utils/QwertyUtils.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "PassDetail.h"

// After this pass finishes, all predications in the program will consist
// only of 1s. This is a crucial step in lowering to a circuit (which, in our
// representation, allows controlling only on 1).

namespace {

// Replace every qwerty.func_pred whose predicate basis is not all 1s with a
// qwerty.lambda that repeatedly calls a 1-predicated version of the
// function, conjugating control qubits as needed.
class FuncPredNotOnesPattern : public mlir::OpRewritePattern<qwerty::FuncPredOp> {
    using mlir::OpRewritePattern<qwerty::FuncPredOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qwerty::FuncPredOp pred,
            mlir::PatternRewriter &rewriter) const override {
        qwerty::BasisAttr basis = pred.getPred();
        if (basis.hasOnlyOnes()) {
            // Already in good shape
            return mlir::failure();
        }
        size_t pred_dim = basis.getDim();

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

            llvm::SmallVector<mlir::Value> pred_qubits(
                qubits_in.begin(),
                qubits_in.begin() + pred_dim);
            llvm::SmallVector<mlir::Value> func_args(
                qubits_in.begin() + pred_dim,
                qubits_in.end());

            // Only create the qwerty.func_pred once, but lazily
            mlir::Value pred_func;

            qwerty::lowerPredBasisToControls(
                rewriter, loc, basis, pred_qubits, pred_qubits,
                [&](llvm::SmallVectorImpl<mlir::Value> &controls) {
                    if (controls.empty()) {
                        // This code will run only once
                        mlir::Value args_packed =
                            rewriter.create<qwerty::QBundlePackOp>(
                                loc, func_args).getQbundle();
                        mlir::ValueRange func_results =
                            rewriter.create<qwerty::CallIndirectOp>(
                                loc, og_func, args_packed).getResults();
                        assert(func_results.size() == 1);
                        mlir::Value func_result = func_results[0];
                        mlir::ValueRange result_unpacked =
                            rewriter.create<qwerty::QBundleUnpackOp>(
                                loc, func_result).getQubits();
                        func_args.clear();
                        func_args.append(result_unpacked.begin(),
                                         result_unpacked.end());
                    } else {
                        size_t controls_dim = controls.size();
                        if (!pred_func) {
                            qwerty::BasisAttr all_ones =
                                qwerty::BasisAttr::getAllOnesBasis(
                                    getContext(), controls_dim);
                            pred_func =
                                rewriter.create<qwerty::FuncPredOp>(
                                    loc, all_ones, og_func).getResult();
                        }

                        llvm::SmallVector<mlir::Value> args(
                            controls.begin(), controls.end());
                        args.append(func_args);
                        mlir::Value args_packed =
                            rewriter.create<qwerty::QBundlePackOp>(
                                loc, args).getQbundle();
                        mlir::ValueRange func_results =
                            rewriter.create<qwerty::CallIndirectOp>(
                                loc, pred_func, args_packed).getResults();
                        assert(func_results.size() == 1);
                        mlir::Value func_result = func_results[0];
                        mlir::ValueRange result_unpacked =
                            rewriter.create<qwerty::QBundleUnpackOp>(
                                loc, func_result).getQubits();

                        controls.clear();
                        controls.append(
                            result_unpacked.begin(),
                            result_unpacked.begin() + controls_dim);
                        func_args.clear();
                        func_args.append(
                            result_unpacked.begin() + controls_dim,
                            result_unpacked.end());
                    }
                });

            llvm::SmallVector<mlir::Value> final_qubits(pred_qubits);
            final_qubits.append(func_args);
            mlir::Value final_packed =
                rewriter.create<qwerty::QBundlePackOp>(
                    loc, final_qubits).getQbundle();
            rewriter.create<qwerty::ReturnOp>(loc, final_packed);
        }

        rewriter.replaceOp(pred, lambda.getResult());
        return mlir::success();
    }
};

// Replace every qwerty.call op with a predicate not consisting of 1s with
// repeated 1-predicated calls, conjugating control qubits as needed.
class CallPredNotOnesPattern : public mlir::OpRewritePattern<qwerty::CallOp> {
    using mlir::OpRewritePattern<qwerty::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
            qwerty::CallOp call,
            mlir::PatternRewriter &rewriter) const override {
        if (!call.getPred()) {
            // Not even a predicated call
            return mlir::failure();
        }
        qwerty::BasisAttr basis = call.getPredAttr();
        if (basis.hasOnlyOnes()) {
            // Already in good shape
            return mlir::failure();
        }
        size_t pred_dim = basis.getDim();

        mlir::Location loc = call.getLoc();
        mlir::ValueRange call_operands = call.getCapturesAndOperands();
        // As is typical with reversible functions, assume that there is
        // exactly one operand and it has type qbundle
        assert(!call_operands.empty());
        mlir::Value qbundle_arg = call_operands[call_operands.size()-1];
        assert(llvm::isa<qwerty::QBundleType>(qbundle_arg.getType()));

        mlir::ValueRange unpacked =
            rewriter.create<qwerty::QBundleUnpackOp>(
                loc, qbundle_arg).getQubits();
        llvm::SmallVector<mlir::Value> pred_qubits(
            unpacked.begin(), unpacked.begin() + pred_dim);
        llvm::SmallVector<mlir::Value> func_args(
            unpacked.begin() + pred_dim,
            unpacked.end());
        qwerty::BasisAttr all_ones_basis;
        mlir::Type new_qbundle_arg_ty;

        qwerty::lowerPredBasisToControls(
            rewriter, loc, basis, pred_qubits, pred_qubits,
            [&](llvm::SmallVectorImpl<mlir::Value> &controls) {
                size_t controls_dim = controls.size();
                if (!new_qbundle_arg_ty) {
                    new_qbundle_arg_ty =
                        rewriter.getType<qwerty::QBundleType>(
                            controls_dim + func_args.size());
                }

                if (!controls_dim) {
                    // This code will run only once
                    mlir::Value args_packed =
                        rewriter.create<qwerty::QBundlePackOp>(
                            loc, func_args).getQbundle();
                    mlir::ValueRange func_results =
                        rewriter.create<qwerty::CallOp>(
                            loc,
                            new_qbundle_arg_ty,
                            call.getCalleeAttr(),
                            call.getAdj(),
                            /*pred=*/nullptr,
                            args_packed).getResults();
                    assert(func_results.size() == 1);
                    mlir::Value func_result = func_results[0];
                    mlir::ValueRange result_unpacked =
                        rewriter.create<qwerty::QBundleUnpackOp>(
                            loc, func_result).getQubits();
                    func_args.clear();
                    func_args.append(result_unpacked.begin(),
                                     result_unpacked.end());
                } else {
                    if (!all_ones_basis) {
                        all_ones_basis =
                            qwerty::BasisAttr::getAllOnesBasis(
                                getContext(), controls_dim);
                    }

                    llvm::SmallVector<mlir::Value> args(
                        controls.begin(), controls.end());
                    args.append(func_args);
                    mlir::Value args_packed =
                        rewriter.create<qwerty::QBundlePackOp>(
                            loc, args).getQbundle();
                    mlir::ValueRange func_results =
                        rewriter.create<qwerty::CallOp>(
                            loc,
                            new_qbundle_arg_ty,
                            call.getCalleeAttr(),
                            call.getAdj(),
                            all_ones_basis,
                            args_packed).getResults();
                    assert(func_results.size() == 1);
                    mlir::Value func_result = func_results[0];
                    mlir::ValueRange result_unpacked =
                        rewriter.create<qwerty::QBundleUnpackOp>(
                            loc, func_result).getQubits();
                    controls.clear();
                    controls.append(
                        result_unpacked.begin(),
                        result_unpacked.begin() + controls_dim);
                    func_args.clear();
                    func_args.append(
                        result_unpacked.begin() + controls_dim,
                        result_unpacked.end());
                }
            });

        llvm::SmallVector<mlir::Value> final_qubits(pred_qubits);
        final_qubits.append(func_args);
        mlir::Value final_packed =
            rewriter.create<qwerty::QBundlePackOp>(
                loc, final_qubits).getQbundle();

        rewriter.replaceOp(call, final_packed);
        return mlir::success();
    }
};

struct OnlyPredOnesPass
        : public qwerty::OnlyPredOnesBase<OnlyPredOnesPass> {
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<FuncPredNotOnesPattern,
                     CallPredNotOnesPattern>(&getContext());

        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
                getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qwerty::createOnlyPredOnesPass() {
    return std::make_unique<OnlyPredOnesPass>();
}
