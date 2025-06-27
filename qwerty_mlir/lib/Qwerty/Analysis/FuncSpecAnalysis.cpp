#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Analysis/FuncSpecAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace qwerty {

void FuncSpec::print(llvm::raw_ostream &os) const {
    os << "(" << sym << "," << (is_adj? "adj" : "fwd") << ","
       << n_controls << ")";
}

FuncSpecSet FuncSpecSet::join(const FuncSpecSet &lhs, const FuncSpecSet &rhs) {
    if (lhs.is_bottom || rhs.is_bottom) {
        return bottom();
    } else {
        SpecSet specs = lhs.specs;
        specs.insert(rhs.specs.begin(), rhs.specs.end());
        return FuncSpecSet(specs);
    }
}

FuncSpecSet FuncSpecSet::adjoint() const {
    if (is_bottom) {
        return bottom();
    }
    FuncSpecSet new_specs;
    for (const FuncSpec &spec : specs) {
        new_specs.specs.insert(spec.adjoint());
    }
    return new_specs;
}

FuncSpecSet FuncSpecSet::control(size_t n_new_controls) const {
    if (is_bottom) {
        return bottom();
    }
    FuncSpecSet new_specs;
    for (const FuncSpec &spec : specs) {
        new_specs.specs.insert(spec.control(n_new_controls));
    }
    return new_specs;
}

void FuncSpecSet::print(llvm::raw_ostream &os) const {
    if (is_bottom) {
        os << "bottom";
    } else {
        os << "{";
        llvm::interleaveComma(specs, os, [&](const FuncSpec &spec) {
            spec.print(os);
        });
        os << "}";
    }
}

mlir::LogicalResult FuncSpecAnalysis::visitOperation(
        mlir::Operation *op,
        llvm::ArrayRef<const FuncSpecLattice *> operands,
        llvm::ArrayRef<FuncSpecLattice *> results) {
    if (qwerty::FuncConstOp func_const =
            llvm::dyn_cast<qwerty::FuncConstOp>(op)) {
        assert(results.size() == 1);
        FuncSpecLattice *result_lattice = results[0];
        propagateIfChanged(
            result_lattice,
            result_lattice->join(
                FuncSpecSet::constant(func_const.getFunc())));
    } else if (qwerty::FuncAdjointOp func_adj =
            llvm::dyn_cast<qwerty::FuncAdjointOp>(op)) {
        assert(operands.size() == 1
               && results.size() == 1);
        const FuncSpecLattice *operand_lattice = operands[0];
        FuncSpecLattice *result_lattice = results[0];
        propagateIfChanged(
            result_lattice,
            result_lattice->join(
                operand_lattice->getValue().adjoint()));
    } else if (qwerty::FuncPredOp func_pred =
            llvm::dyn_cast<qwerty::FuncPredOp>(op)) {
        assert(operands.size() == 1
               && results.size() == 1);
        const FuncSpecLattice *operand_lattice = operands[0];
        FuncSpecLattice *result_lattice = results[0];
        propagateIfChanged(
            result_lattice,
            result_lattice->join(
                operand_lattice->getValue().control(
                    func_pred.getPred().getDim())));
    } else {
        // bottom
        setAllToEntryStates(results);
    }
    return mlir::success();
}

} // namespace qwerty
