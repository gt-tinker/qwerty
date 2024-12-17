#include "llvm/Support/Debug.h"

#include "Qwerty/Analysis/QubitIndexAnalysis.h"

#include "QCirc/Utils/QCircUtils.h"
#include "Qwerty/IR/QwertyOps.h"

#define DEBUG_TYPE "qubit-idx"

namespace {
using Analysis = qwerty::QubitIndexAnalysis;

void initializeBlockArgs(llvm::SmallVectorImpl<mlir::Value> &block_args,
                         mlir::Block &block,
                         size_t &next_qubit_idx,
                         Analysis &analysis) {
    for (auto [i, block_arg] : llvm::enumerate(block_args)) {
        qwerty::QubitIndexVec indices;

        if (qcirc::NonStationaryTypeInterface nonstat =
                llvm::dyn_cast<qcirc::NonStationaryTypeInterface>(
                    block_arg.getType())) {
            for (size_t i = 0; i < nonstat.getNumQubits(); i++) {
                indices.indices.push_back(next_qubit_idx++);
            }
        } else {
            indices = qwerty::QubitIndexVec::bottom();
        }

        analysis[block_arg] = indices;

        LLVM_DEBUG({
            llvm::dbgs() << "assigned indices ";
            indices.print(llvm::dbgs());
            llvm::dbgs() << " to block argument #" << i << ": ";
            block_arg.dump();
        });
    }
}

void visitOperation(mlir::Operation *op, size_t &next_qubit_idx, Analysis &analysis) {
    if (op->hasTrait<qcirc::IsStationaryOpTrait>()) {
        for (mlir::Value result : op->getResults()) {
            analysis[result] = qwerty::QubitIndexVec::bottom();
        }
        LLVM_DEBUG({
            llvm::dbgs() << "stationary, setting results to bottoms: ";
            op->dump();
        });
    } else if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        // Do nothing.
        LLVM_DEBUG({
            llvm::dbgs() << "ignoring terminator: ";
            op->dump();
        });
    } else if (llvm::isa<qcirc::AdjointableOpInterface>(op)) {
        // qballoc/qbprep case
        if (!op->getNumOperands()) {
            for (mlir::Value result : op->getResults()) {
                qwerty::QubitIndexVec indices;

                if (qcirc::NonStationaryTypeInterface nonstat =
                        llvm::dyn_cast<qcirc::NonStationaryTypeInterface>(result.getType())) {
                    for (size_t i = 0; i < nonstat.getNumQubits(); i++) {
                        indices.indices.push_back(next_qubit_idx++);
                    }
                } else {
                    indices = qwerty::QubitIndexVec::bottom();
                }

                analysis[result] = indices;

                LLVM_DEBUG({
                    llvm::dbgs() << "assigned indices ";
                    indices.print(llvm::dbgs());
                    llvm::dbgs() << " to ";
                    result.dump();
                });
            }
        } else if (!op->getNumResults()) {
            // Hopefully this is a freez/discardz. Nothing to do.
            LLVM_DEBUG({
                llvm::dbgs() << "ignoring sink node: ";
                op->dump();
            });
        } else {
            llvm::SmallVector<mlir::Value> nonstat_operands;
            qcirc::AdjointableOpInterface adjointable =
                llvm::cast<qcirc::AdjointableOpInterface>(op);
            for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
                if (!adjointable.isStationaryOperand(i)) {
                    assert(analysis.count(operand) && "operand missing from analysis");
                    nonstat_operands.push_back(operand);
                }
            }

            if (nonstat_operands.size() == op->getNumResults()) {
                for (auto [nonstat_operand, result] : llvm::zip(nonstat_operands,
                                                                op->getResults())) {
                    assert(analysis.count(nonstat_operand)
                           && "nonstat operand missing from analysis");
                    // This is an intentional copy. We can't safely use a
                    // reference because analysis[result] may cause the
                    // DenseMap to be resized, invalidating our reference
                    qwerty::QubitIndexVec inds = analysis[nonstat_operand];
                    assert(!inds.isBottom());
                    analysis[result] = inds;

                    LLVM_DEBUG({
                        llvm::dbgs() << "assigned indices ";
                        inds.print(llvm::dbgs());
                        llvm::dbgs() << " to ";
                        result.dump();
                    });
                }
            } else if (nonstat_operands.size() < op->getNumResults()) {
                // "For now"
                assert(nonstat_operands.size() == 1);

                // This is an intentional copy. We can't safely use a
                // reference because analysis[result] below may cause the
                // DenseMap to be resized, invalidating our reference
                qwerty::QubitIndexVec inds = analysis[nonstat_operands[0]];
                assert(!inds.isBottom());
                assert(op->getNumResults() == inds.indices.size());
                // In this case, we have an op like qbunpack, which has an array of qubit indices associated with one MLIR value.
                for (auto [i, result] : llvm::enumerate(op->getResults())) {
                    analysis[result] = inds.indices[i];
                    LLVM_DEBUG({
                        llvm::dbgs() << "assigned indices ";
                        analysis[result].print(llvm::dbgs());
                        llvm::dbgs() << " to ";
                        result.dump();
                    });
                }
            } else {
                assert(op->getNumResults() == 1);
                mlir::Value result = op->getResult(0);

                qwerty::QubitIndexVec new_inds;
                for (mlir::Value nonstat : nonstat_operands) {
                    const qwerty::QubitIndexVec &inds = analysis[nonstat];
                    assert(!inds.isBottom());
                    new_inds.indices.append(inds.indices.begin(), inds.indices.end());
                }
                analysis[result] = new_inds;

                LLVM_DEBUG({
                    llvm::dbgs() << "assigned indices ";
                    new_inds.print(llvm::dbgs());
                    llvm::dbgs() << " to ";
                    result.dump();
                });
            }
        }
    } else {
        assert(0 && "Unsupported op in reversible basic block"); // (should be) unreachable
    }
}
} // namespace

namespace qwerty {

void QubitIndexVec::print(llvm::raw_ostream &os) const {
    if (isBottom()) {
        os << "bottom";
    } else {
        os << '[';
        for (size_t i = 0; i < indices.size(); ++i) {
            os << indices[i] << (i < indices.size() - 1 ? "," : "");
        }
        os << ']';
    }
}

Analysis runQubitIndexAnalysis(mlir::Block &block,
                               mlir::Operation *start_at,
                               llvm::SmallVectorImpl<mlir::Value> &block_args) {
    size_t next_qubit_idx = 0;
    Analysis analysis;

    initializeBlockArgs(block_args, block, next_qubit_idx, analysis);

    for (mlir::Operation *op = start_at; op; op = op->getNextNode()) {
        visitOperation(op, next_qubit_idx, analysis);
    }

    return analysis;
}

Analysis runQubitIndexAnalysis(mlir::Block &block) {
    llvm::SmallVector<mlir::Value> actual_block_args(block.getArguments());
    mlir::Operation *first_op = &block.front();
    return runQubitIndexAnalysis(block, first_op, actual_block_args);
}

} // namespace qwerty
