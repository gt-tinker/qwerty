#ifndef DIALECT_INCLUDE_QWERTY_ANALYSIS_QUBIT_INDEX_H
#define DIALECT_INCLUDE_QWERTY_ANALYSIS_QUBIT_INDEX_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/IRMapping.h"

// Analysis identifying which qubit indices a qubit mlir::Value corresponds to
// (Section 5.3 of the CGO paper). Originally, this used the MLIR dataflow
// framework, but now it is a handrolled for loop (and a whole lot simpler).

namespace qwerty {

// The list of qubit indices that correspond to an MLIR value. For a
// qcirc.qubit, this would be one index. For a qwerty.qbundle[N], this would be
// N indices.
struct QubitIndexVec {
    using Indices = llvm::SmallVector<size_t>;
    Indices indices;
    bool bottomed = false;

    QubitIndexVec() = default;
    QubitIndexVec(size_t index) {
        indices.push_back(index);
    }

    static QubitIndexVec bottom() {
        QubitIndexVec bvec;
        bvec.bottomed = true;
        return bvec;
    }

    bool isBottom() const {
        return bottomed;
    }

    void print(llvm::raw_ostream &os) const;
};

using QubitIndexAnalysis = llvm::DenseMap<mlir::Value, QubitIndexVec>;

QubitIndexAnalysis runQubitIndexAnalysis(mlir::Block &block);
QubitIndexAnalysis runQubitIndexAnalysis(mlir::Block &block,
                                         mlir::Operation *start_at,
                                         llvm::SmallVectorImpl<mlir::Value> &block_args);

} // namespace qwerty

#endif
