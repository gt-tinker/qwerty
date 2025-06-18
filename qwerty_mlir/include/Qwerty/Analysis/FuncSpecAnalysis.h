#ifndef DIALECT_INCLUDE_QWERTY_ANALYSIS_FUNCSPEC_H
#define DIALECT_INCLUDE_QWERTY_ANALYSIS_FUNCSPEC_H

#include "llvm/ADT/SmallSet.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

// Analysis for finding necessary functional specializations for a Qwerty
// program (Section 6.2 of the CGO paper). This uses the MLIR dataflow
// framework.

namespace qwerty {

// A particular specialization of a function
struct FuncSpec {
    std::string sym;
    bool is_adj;
    size_t n_controls;

    FuncSpec(llvm::StringRef sym_ref, bool is_adj, size_t n_controls)
            : sym(sym_ref.str()), is_adj(is_adj), n_controls(n_controls) {}

    FuncSpec adjoint() const {
        return FuncSpec(sym, !is_adj, n_controls);
    }

    FuncSpec control(size_t n_new_controls) const {
        return FuncSpec(sym, is_adj, n_controls + n_new_controls);
    }

    bool operator==(const FuncSpec &other) const {
        return other.sym == sym
               && other.is_adj == is_adj
               && other.n_controls == n_controls;
    }

    bool operator<(const FuncSpec &other) const {
        return sym < other.sym
               || (sym == other.sym && static_cast<int>(is_adj) < static_cast<int>(other.is_adj))
               || (sym == other.sym && is_adj == other.is_adj && n_controls < other.n_controls);
    }

    void print(llvm::raw_ostream &os) const;
};

// A set of function specializations that some function value could
// conservatively point to
struct FuncSpecSet {
    using SpecSet = llvm::SmallSet<FuncSpec, 4>;
    bool is_bottom;
    SpecSet specs;

    FuncSpecSet() : is_bottom(false) {}
    FuncSpecSet(bool is_bottom) : is_bottom(is_bottom) {}
    FuncSpecSet(const SpecSet &specs) : is_bottom(false), specs(specs) {}
    FuncSpecSet(llvm::StringRef sym, bool is_adj, size_t n_controls)
               : is_bottom(false) {
        specs.insert(FuncSpec(sym, is_adj, n_controls));
    }

    static FuncSpecSet join(const FuncSpecSet &lhs, const FuncSpecSet &rhs);

    static FuncSpecSet bottom() {
        return FuncSpecSet(/*is_bottom=*/true);
    }

    static FuncSpecSet entryState() {
        return bottom();
    }

    static FuncSpecSet constant(llvm::StringRef sym) {
        return FuncSpecSet(sym, /*is_adj=*/false, /*n_controls=*/0);
    }

    FuncSpecSet adjoint() const;

    FuncSpecSet control(size_t n_new_controls) const;

    bool operator==(const FuncSpecSet &other) const {
        return (other.is_bottom && is_bottom)
               || (!other.is_bottom && !is_bottom && other.specs == specs);
    }

    void print(llvm::raw_ostream &os) const;
};

// The rest of this is required by the MLIR dataflow framework. For some
// out-of-date documentation see:
// https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/

class FuncSpecLattice : public mlir::dataflow::Lattice<FuncSpecSet> {
public:
    using Lattice::Lattice;
};

class FuncSpecAnalysis : public mlir::dataflow::SparseForwardDataFlowAnalysis<FuncSpecLattice> {
public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    void setToEntryState(FuncSpecLattice *lattice) override {
        propagateIfChanged(lattice,
                           lattice->join(FuncSpecSet::entryState()));
    }

    mlir::LogicalResult visitOperation(mlir::Operation *op,
                        llvm::ArrayRef<const FuncSpecLattice *> operands,
                        llvm::ArrayRef<FuncSpecLattice *> results) override;
};

} // namespace qwerty

#endif // DIALECT_INCLUDE_QWERTY_ANALYSIS_FUNCSPEC_H
