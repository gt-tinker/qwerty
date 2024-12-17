#include <queue>
#include <unordered_set>

#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

#include "QCirc/IR/QCircInterfaces.h"
#include "QCirc/Utils/QCircUtils.h"

// This is the code that takes the adjoint of a basic block (Section 5.2 of
// the CGO paper).

// For LLVM debug macros:
// https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option
#define DEBUG_TYPE "adjoint"

namespace {

/////// Helpers for manipulating/exploring ops ///////

// Check for ops that cannot have stationary operands. Used to skip thorny code
// that does not cooperate with such ops
bool cannotHaveStatOperands(mlir::Operation *op) {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        // Terminators have no stationary operands
        return true;
    }

    qcirc::AdjointableOpInterface adj = llvm::cast<qcirc::AdjointableOpInterface>(op);
    // Sources and sinks are not allowed to have stationary operands. Why?
    // Well, if the adjoint form has more stationary operands, where would
    // those come from?
    if (adj.getNumOperandsOfAdjoint() != op->getNumOperands()) {
        return true;
    }

    return false;
}

// Return the total number of operands (both stationary and nonstationary) this
// operation will have once it is adjointed.
size_t getAdjNumTotalOperands(mlir::Operation *op) {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return op->getNumOperands();
    }
    qcirc::AdjointableOpInterface adj = llvm::cast<qcirc::AdjointableOpInterface>(op);
    return adj.getNumOperandsOfAdjoint();
}

// Return the number of nonstationary operands this operation will have once it
// is adjointed.
size_t getAdjNumNonstatOperands(mlir::Operation *op) {
    if (cannotHaveStatOperands(op)) {
        return getAdjNumTotalOperands(op);
    }

    qcirc::AdjointableOpInterface adj = llvm::cast<qcirc::AdjointableOpInterface>(op);

    size_t total = 0;
    for (size_t i = 0; i < op->getNumOperands(); i++) {
        if (!adj.isStationaryOperand(i)) {
            total++;
        }
    }
    return total;
}

// Return the number of stationary operands this operation will have once it is
// adjointed.
size_t getAdjNumStatOperands(mlir::Operation *op) {
    return getAdjNumTotalOperands(op) - getAdjNumNonstatOperands(op);
}

// Return all of the nonstationary operands on the _forward_ (current) version
// of op.
void getFwdNonstatOperands(
        mlir::Operation *op,
        llvm::SmallVectorImpl<mlir::Value> &nonstat_fwd_operands) {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return nonstat_fwd_operands.append(
            op->operand_begin(), op->operand_end());
    }
    qcirc::AdjointableOpInterface adj = llvm::cast<qcirc::AdjointableOpInterface>(op);
    nonstat_fwd_operands.clear();
    for (size_t i = 0; i < op->getNumOperands(); i++) {
        if (!adj.isStationaryOperand(i)) {
            nonstat_fwd_operands.push_back(op->getOperand(i));
        }
    }
}

mlir::LogicalResult reshapeBlockArgs(
        mlir::RewriterBase &rewriter,
        qcirc::Reshaper &reshaper,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        mlir::IRMapping &stationary_vals,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &adj_block_args_out) {
    llvm::SmallVector<mlir::Value> nonstat_block_args;
    bool saw_nonstat = false;
    for (mlir::Value arg : block_args) {
        if (llvm::isa<qcirc::NonStationaryTypeInterface>(
                arg.getType())) {
            // Is nonstationary (e.g., a qubit or qbundle)
            nonstat_block_args.push_back(arg);
            saw_nonstat = true;
        } else {
            // Is stationary (e.g., a float param or a function)
            assert(!saw_nonstat
                   && "Cannot intermix stationary and nontstationary block "
                      "args");
            stationary_vals.map(arg, arg);
        }
    }

    mlir::Operation *fwd_term = fwd_block.getTerminator();
    mlir::ValueTypeRange<llvm::SmallVector<mlir::Value>>
        nonstat_block_arg_types(nonstat_block_args);
    if (fwd_term->getOperandTypes() == nonstat_block_arg_types) {
        // Happy path: types match up, no reshaping needed
        adj_block_args_out = nonstat_block_args;
        return mlir::success();
    } else {
        return reshaper.reshape(rewriter,
                                nonstat_block_args,
                                fwd_term->getOperandTypes(),
                                adj_block_args_out);
    }
}

// Return the result mlir::Values of the adjoint version of op
void getAdjResults(
        mlir::Operation *op,
        mlir::RewriterBase &rewriter,
        llvm::SmallVectorImpl<mlir::Value> &adj_block_args,
        llvm::SmallVectorImpl<mlir::Value> &adj_inputs,
        llvm::SmallVectorImpl<mlir::Value> &adj_outputs) {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        // Special case: treat arguments of new adjoint block as outputs of the
        // "adjointed" forward terminator
        adj_outputs = adj_block_args;
        return;
    }

    qcirc::AdjointableOpInterface adj = llvm::cast<qcirc::AdjointableOpInterface>(op);
    adj.buildAdjoint(rewriter, adj_inputs, adj_outputs);
}

/////// Code driving the DAG traversal and re-creation ///////

struct PendingAdjOp {
    mlir::Operation *op;
    llvm::SmallVector<mlir::Value> adj_operands;
    size_t num_stat_operands;
    size_t num_adj_operands_left;

    PendingAdjOp(mlir::Operation *op)
        : op(op),
          adj_operands(getAdjNumTotalOperands(op)),
          num_stat_operands(getAdjNumStatOperands(op)),
          num_adj_operands_left(getAdjNumNonstatOperands(op)) {}

    PendingAdjOp(size_t num_operands)
        : op(nullptr),
          adj_operands(num_operands),
          num_stat_operands(0),
          num_adj_operands_left(num_operands) {}
};

void checkBlockOpsAndInitQueue(mlir::Block &fwd_block,
                               std::unordered_set<mlir::Operation *> &stray_ops,
                               std::queue<PendingAdjOp> &queue) {
    // Iterate in reverse so we hit the terminator first. Why? It makes me feel
    // warmer and fuzzier than pushing a bunch of qfrees first, although either
    // way should work in theory.
    for (mlir::Operation &fwd_op : llvm::reverse(fwd_block)) {
        // Initially assume that every operation is unreachable via walking the
        // dag backwards from the terminator. We'll remove ops from this set as
        // we encounter them
        [[maybe_unused]] bool inserted = stray_ops.insert(&fwd_op).second;
        assert(inserted && "Encountered the same op twice??");

        if (fwd_op.hasTrait<mlir::OpTrait::IsTerminator>()) {
            // Start our reverse traversal at the forward terminator
            queue.emplace(&fwd_op);
        } else if (fwd_op.hasTrait<qcirc::IsStationaryOpTrait>()) {
            // We will clone this below if it's needed
        } else if (qcirc::AdjointableOpInterface adj =
                    llvm::dyn_cast<qcirc::AdjointableOpInterface>(&fwd_op)) {
            assert(adj.isAdjointable() && "Non-adjointable operation found");
            if (!adj.getNumOperandsOfAdjoint()) {
                // We will never find this walking backwards from the forward
                // terminator, so enqueue it manually
                queue.emplace(&fwd_op);
            } else {
                // We will find this below and adjoint it as needed
            }
        } else {
            assert(0 && "Forward block contains non-adjointable operation");
        }
    }
}

mlir::Value cloneStationaryValue(mlir::Value fwd_operand,
                                 mlir::OpBuilder &builder,
                                 mlir::IRMapping &stationary_vals) {
    if (!stationary_vals.contains(fwd_operand)) {
        mlir::Operation *op = fwd_operand.getDefiningOp();
        assert(op);
        assert(op->hasTrait<qcirc::IsStationaryOpTrait>());

        // Clone all operands if needed
        for (mlir::Value operand : op->getOperands()) {
            cloneStationaryValue(operand, builder, stationary_vals);
        }

        // Will automatically insert results of new op into the mapper
        builder.clone(*op, stationary_vals);
    }

    return stationary_vals.lookup(fwd_operand);
}

void populateStationaryValues(
        mlir::Operation *op,
        mlir::OpBuilder &builder,
        llvm::SmallVectorImpl<mlir::Value> &adj_operands,
        mlir::IRMapping &stationary_vals) {
    // The code below gets messy with sources and sinks. Don't bother.
    if (cannotHaveStatOperands(op)) {
        return;
    }

    assert(op->getNumOperands() == adj_operands.size());
    for (auto [i, fwd_operand, adj_operand] :
            llvm::enumerate(op->getOperands(), adj_operands)) {
        if (!adj_operand) {
            adj_operands[i] = cloneStationaryValue(fwd_operand,
                                                   builder,
                                                   stationary_vals);
        }
    }
}

void propagateAdjResultAcrossOpResult(
        mlir::OpResult op_res,
        llvm::DenseMap<mlir::Operation *, PendingAdjOp> &staging,
        PendingAdjOp **pending_out,
        size_t *idx_out) {
    mlir::Operation *parent = op_res.getOwner();
    auto it = staging.find(parent);
    if (it == staging.end()) {
        auto [new_it, added] =
            staging.try_emplace(parent, parent);
        assert(added);
        *pending_out = &new_it->getSecond();
    } else {
        *pending_out = &it->getSecond();
    }
    *idx_out = op_res.getResultNumber();
}

void propagateAdjResults(
        mlir::Operation *op,
        llvm::SmallDenseMap<mlir::Value, size_t> &adj_arg_indices,
        llvm::SmallVectorImpl<mlir::Value> &adj_outputs,
        PendingAdjOp &block_args_dummy,
        std::queue<PendingAdjOp> &queue,
        llvm::DenseMap<mlir::Operation *, PendingAdjOp> &staging) {

    llvm::SmallVector<mlir::Value> nonstat_fwd_operands;
    getFwdNonstatOperands(op, nonstat_fwd_operands);
    assert(nonstat_fwd_operands.size() == adj_outputs.size());

    for (auto [fwd_operand, adj_result] :
            llvm::zip(nonstat_fwd_operands, adj_outputs)) {
        PendingAdjOp *pending;
        size_t idx;
        if (adj_arg_indices.count(fwd_operand)) {
            pending = &block_args_dummy;
            idx = adj_arg_indices.at(fwd_operand);
        } else if (mlir::OpResult op_res =
                llvm::dyn_cast<mlir::OpResult>(fwd_operand)) {
            propagateAdjResultAcrossOpResult(op_res, staging, &pending, &idx);
        } else {
            assert(0 && "Mystery mlir::Value");
            pending = nullptr;
        }

        // Stationary operands are not returned. So to recover the operand number
        // from a result number, we have to offset the result number by the number
        // of stationary operands.
        idx += pending->num_stat_operands;

        assert(!pending->adj_operands[idx]);
        pending->adj_operands[idx] = adj_result;
        assert(pending->num_adj_operands_left);
        if (!--pending->num_adj_operands_left) {
            mlir::Operation *op = pending->op;
            // op is NULL for the dummy block argument PendingAdjOp
            if (op) {
                queue.push(std::move(*pending));
                staging.erase(op);
            }
        }
    }
}

void eraseForwardOps(mlir::RewriterBase &rewriter,
                     llvm::SmallVectorImpl<mlir::Operation *> &pending_erase,
                     std::unordered_set<mlir::Operation *> &stray_ops) {
    // Small optimization: we stored ops we encountered in the dag traversal
    // already in reverse topological order. Now we can just delete them
    for (mlir::Operation *op : pending_erase) {
        rewriter.eraseOp(op);
    }

    llvm::SmallVector<mlir::Operation *> ops(stray_ops.begin(),
                                             stray_ops.end());
    mlir::computeTopologicalSorting(ops);
    for (mlir::Operation *op : llvm::reverse(ops)) {
        assert(mlir::isMemoryEffectFree(op)
               && "Stray operation is not accessible by dag, but I cannot "
                  "delete it because it is not pure. This situation is not "
                  "supported.");
        rewriter.eraseOp(op);
    }
}

} // namespace

namespace qcirc {

mlir::LogicalResult takeAdjointOfBlockInPlaceNoTerm(
        mlir::RewriterBase &rewriter,
        Reshaper &reshaper,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &term_args) {
    llvm::SmallVector<mlir::Value> actual_block_args(fwd_block.getArguments());
    return takeAdjointOfBlockInPlaceNoTerm(rewriter, reshaper, fwd_block,
                                           actual_block_args, term_args);
}

mlir::LogicalResult takeAdjointOfBlockInPlaceNoTerm(
        mlir::RewriterBase &rewriter,
        Reshaper &reshaper,
        mlir::Block &fwd_block,
        llvm::SmallVectorImpl<mlir::Value> &block_args,
        llvm::SmallVectorImpl<mlir::Value> &term_args) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    // Ops that we did not reach in our backwards dag traversal
    std::unordered_set<mlir::Operation *> stray_ops;
    llvm::SmallVector<mlir::Operation *> pending_erase;
    llvm::DenseMap<mlir::Operation *, PendingAdjOp> staging;
    std::queue<PendingAdjOp> queue;
    mlir::IRMapping stationary_vals;

    LLVM_DEBUG({
        llvm::dbgs() << "before:\n";
        fwd_block.print(llvm::dbgs());
    });

    checkBlockOpsAndInitQueue(fwd_block, stray_ops, queue);

    rewriter.setInsertionPointToStart(&fwd_block);
    llvm::SmallVector<mlir::Value> adj_block_args;
    if (reshapeBlockArgs(rewriter, reshaper, block_args,
                         stationary_vals, fwd_block,
                         adj_block_args).failed()) {
        return mlir::failure();
    }
    llvm::SmallDenseMap<mlir::Value, size_t> adj_arg_indices;
    for (auto [i, arg] : llvm::enumerate(adj_block_args)) {
        adj_arg_indices.insert({arg, i});
    }

    PendingAdjOp block_args_dummy(adj_block_args.size());

    while (!queue.empty()) {
        PendingAdjOp &next = queue.front();

        populateStationaryValues(next.op, rewriter, next.adj_operands,
                                 stationary_vals);

        llvm::SmallVector<mlir::Value> adj_outputs;
        getAdjResults(next.op, rewriter, adj_block_args, next.adj_operands,
                      adj_outputs);

        propagateAdjResults(next.op, adj_arg_indices, adj_outputs,
                            block_args_dummy, queue, staging);

        if (next.op) {
            LLVM_DEBUG({
                llvm::dbgs() << "popped:\n";
                next.op->print(llvm::dbgs());
                llvm::dbgs() << "\n";
            });
            // We have encountered this op, so it can't be a stray
            stray_ops.erase(next.op);
            // Queue up this op for erasure. Why can't we delete it right now?
            // A disturbing reason: it could be our insertion point, which
            // would cause a segfault when we try to insert something next.
            // Best we can do (safely) is to queue up all the ops to delete in
            // reverse topological order
            pending_erase.push_back(next.op);
        }

        // Now that we're done free the memory
        queue.pop();
    }

    LLVM_DEBUG({
        llvm::dbgs() << "after:\n";
        fwd_block.print(llvm::dbgs());
    });
    assert(staging.empty() && "Ops starved");

    eraseForwardOps(rewriter, pending_erase, stray_ops);

    // It is takeAdjointOfBlockInPlace()'s job to actually build the terminator
    term_args.clear();
    term_args.append(block_args_dummy.adj_operands.begin(),
                     block_args_dummy.adj_operands.end());
    return mlir::success();
}

} // namespace qcirc
