// This code is taken from transform_synth.cpp in Tweedledum. The following is
// their original comment:
//
// > This implementation is based on:
// >
// > Miller, D. Michael, Dmitri Maslov, and Gerhard W. Dueck. "A transformation
// > based algorithm for reversible logic synthesis." Proceedings 2003. design
// > automation conference (ieee cat. no. 03ch37451). IEEE, 2003.
// >
// > Starting from a reversible function, transformation-based synthesis applies
// > gates and adjusts the function representation accordingly in a way that each
// > gate application gets the function closer to the identity function.  If the
// > identity function has been reached, all applied gates make up for the circuit
// > that realizes the initial function.
// >
// > Here there is also the implementation of a multidirectional method based on:
// >
// > Soeken, Mathias, Gerhard W. Dueck, and D. Michael Miller. "A fast symbolic
// > transformation based algorithm for reversible logic synthesis." International
// > Conference on Reversible Computation. Springer, Cham, 2016.
// >
// > Variants:
// > (*) unidirectional: only adds gates from the output side
// > (*) bidirectional: adds gates from input __or__ output side at each step
// > (*) multidirectional: adds gates from input __and__ output side at each step
//
// However, we have removed the unidirectional and bidirectional code.

#include "util.hpp"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"

namespace {

using AbstractGate = std::pair<uint32_t, uint32_t>;
using GateList = std::vector<AbstractGate>;

inline void update_permutation(
  std::vector<uint32_t>& perm, uint32_t controls, uint32_t targets)
{
    for (uint32_t i = 0; i < perm.size(); ++i) {
        if ((perm[i] & controls) == controls) {
            perm[i] ^= targets;
        }
    }
}

inline void update_permutation_inv(
  std::vector<uint32_t>& perm, uint32_t controls, uint32_t targets)
{
    for (uint32_t i = 0u; i < perm.size(); ++i) {
        if ((i & controls) != controls) {
            continue;
        }
        uint32_t const partner = i ^ targets;
        if (partner > i) {
            std::swap(perm[i], perm[partner]);
        }
    }
}

inline GateList multidirectional(std::vector<uint32_t> perm)
{
    GateList gates;
    auto pos = gates.begin();
    for (uint32_t i = 0u; i < perm.size(); ++i) {
        // Find cheapest assignment
        uint32_t x_best = i;
        uint32_t x_best_cost = BITS_POPCOUNT(i ^ perm[i]);
        for (uint32_t j = i + 1; j < perm.size(); ++j) {
            uint32_t j_cost = BITS_POPCOUNT(i ^ perm[j]);
            uint32_t cost = BITS_POPCOUNT(i ^ j) + j_cost;
            if (cost < x_best_cost) {
                x_best = j;
                x_best_cost = cost;
            }
        }

        uint32_t const y = perm[x_best];
        // map x |-> i
        uint32_t p = ~x_best & i;
        if (p) {
            update_permutation_inv(perm, x_best, p);
            pos = gates.emplace(pos, x_best, p);
            pos++;
        }
        uint32_t q = x_best & ~i;
        if (q) {
            update_permutation_inv(perm, i, q);
            pos = gates.emplace(pos, i, q);
            pos++;
        }

        // map y |-> i
        p = i & ~y;
        if (p) {
            update_permutation(perm, y, p);
            pos = gates.emplace(pos, y, p);
        }
        q = ~i & y;
        if (q) {
            update_permutation(perm, i, q);
            pos = gates.emplace(pos, i, q);
        }
    }
    return gates;
}

} // namespace

namespace qcirc {

void synthPermutationSlow(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx,
        const std::vector<uint32_t> &perm) {
    GateList gates = multidirectional(perm);
    for (auto [controls, targets] : gates) {
        llvm::SmallVector<mlir::Value> this_control_qubits(
            control_qubits.begin(), control_qubits.end());
        for (uint32_t c = 0, ctrls = controls; ctrls; ctrls >>= 1, ++c) {
            if (ctrls & 1) {
                this_control_qubits.push_back(qubits[qubit_idx + c]);
            }
        }
        for (uint32_t t = 0; targets; targets >>= 1, ++t) {
            if ((targets & 1) == 0) {
                continue;
            }

            qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, this_control_qubits, qubits[qubit_idx + t]);
            mlir::ValueRange control_results = x.getControlResults();
            this_control_qubits.clear();
            this_control_qubits.append(control_results.begin(),
                                       control_results.end());
            qubits[qubit_idx + t] = x.getResult();
        }

        size_t nonfixed_ctrls_start = control_qubits.size();
        control_qubits.clear();
        control_qubits.append(this_control_qubits.begin(),
                              this_control_qubits.begin() + nonfixed_ctrls_start);
        for (uint32_t c = 0; controls; controls >>= 1, ++c) {
            if (controls & 1) {
                qubits[qubit_idx + c] = this_control_qubits[nonfixed_ctrls_start++];
            }
        }
    }
}

// From my and Pulkit's brain, not Tweedledum
void synthPermutationFast(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        size_t qubit_idx,
        const llvm::SmallVector<std::pair<llvm::APInt, llvm::APInt>> &perm) {
    // Mapping of XOR masks to ancilla indices and lists of vectors that need
    // this mask applied
    llvm::DenseMap<llvm::APInt,
                   std::pair<size_t, llvm::SmallVector<llvm::APInt>>> mask_vecs;
    // Pairs of ancilla indices and controls to clean up
    llvm::SmallVector<std::pair<size_t, llvm::APInt>> cleanup_queue;

    for (const std::pair<llvm::APInt, llvm::APInt> &perm_pair : perm) {
        const llvm::APInt &left = perm_pair.first;
        const llvm::APInt &right = perm_pair.second;
        llvm::APInt mask(left);
        mask ^= right;

        if (!mask.isZero()) {
            auto result_pair = mask_vecs.try_emplace(mask,
                mask_vecs.size(), std::initializer_list<llvm::APInt>{});
            result_pair.first->getSecond().second.push_back(left);
            size_t ancilla_idx = result_pair.first->getSecond().first;
            cleanup_queue.emplace_back(ancilla_idx, right);
        }
    }

    // Will be useful in a second for iterating over the map deterministically
    llvm::SmallVector<llvm::APInt> sorted_masks;
    for (auto mask_vecs_entry : mask_vecs) {
        sorted_masks.push_back(mask_vecs_entry.getFirst());
    }
    llvm::sort(sorted_masks, [](const llvm::APInt &left, const llvm::APInt &right) {
        return left.ult(right);
    });

    // Allocate ancilla
    llvm::SmallVector<mlir::Value> ancillas;
    for (size_t i = 0; i < mask_vecs.size(); i++) {
        ancillas.push_back(builder.create<qcirc::QallocOp>(loc).getResult());
    }

    // Flip ancilla bits in subspaces
    for (llvm::APInt &mask : sorted_masks) {
        auto mask_vecs_pair = mask_vecs.at(mask);
        size_t ancilla_idx = mask_vecs_pair.first;
        llvm::SmallVector<llvm::APInt> &vecs = mask_vecs_pair.second;

        // TODO: optimize the case where e.g. a C0NOT and C1NOT can be replaced
        //       with a NOT
        mlir::Value ancilla = ancillas[ancilla_idx];

        for (llvm::APInt &vec : vecs) {
            llvm::SmallVector<mlir::Value> controls(control_qubits.begin(),
                                                    control_qubits.end());
            for (size_t j = 0; j < vec.getBitWidth(); j++) {
                mlir::Value ctrl = qubits[qubit_idx + j];
                bool bit = vec[vec.getBitWidth()-1-j];
                if (!bit) {
                    // Controlled-on-zero
                    ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                        ctrl).getResult();
                }
                controls.push_back(ctrl);
            }

            qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, controls, ancilla);
            ancilla = x.getResult();

            mlir::ValueRange control_results = x.getControlResults();
            assert(control_results.size() == controls.size()
                   && "wrong number of output control qubits");
            size_t n_fixed_controls = control_qubits.size();
            control_qubits.clear();
            control_qubits.append(control_results.begin(),
                                  control_results.begin() + n_fixed_controls);

            for (size_t j = 0; j < vec.getBitWidth(); j++) {
                mlir::Value ctrl = control_results[n_fixed_controls + j];
                bool bit = vec[vec.getBitWidth()-1-j];
                if (!bit) {
                    // Undo bit flip above
                    ctrl = builder.create<qcirc::Gate1QOp>(
                        loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                        ctrl).getResult();
                }
                qubits[qubit_idx + j] = ctrl;
            }
        }

        ancillas[ancilla_idx] = ancilla;
    }

    // Flip main qubits based on ancilla
    for (llvm::APInt &mask : sorted_masks) {
        auto mask_vecs_pair = mask_vecs.at(mask);
        size_t ancilla_idx = mask_vecs_pair.first;
        mlir::Value ancilla = ancillas[ancilla_idx];

        for (size_t j = 0; j < mask.getBitWidth(); j++) {
            bool bit = mask[mask.getBitWidth()-1-j];
            if (bit) {
                qcirc::Gate1QOp cnot = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, ancilla, qubits[qubit_idx + j]);
                assert(cnot.getControlResults().size() == 1
                       && "expected one control result from cnot");
                ancilla = cnot.getControlResults()[0];
                qubits[qubit_idx + j] = cnot.getResult();
            }
        }

        ancillas[ancilla_idx] = ancilla;
    }

    // Uncompute (clean up ancilla)
    for (auto [ancilla_idx, vec] : cleanup_queue) {
        // TODO: optimize the case where e.g. a C0NOT and C1NOT can be replaced
        //       with a NOT
        mlir::Value ancilla = ancillas[ancilla_idx];

        llvm::SmallVector<mlir::Value> controls(control_qubits.begin(),
                                                control_qubits.end());
        for (size_t j = 0; j < vec.getBitWidth(); j++) {
            mlir::Value ctrl = qubits[qubit_idx + j];
            bool bit = vec[vec.getBitWidth()-1-j];
            if (!bit) {
                // Controlled-on-zero
                ctrl = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                    ctrl).getResult();
            }
            controls.push_back(ctrl);
        }

        qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::X, controls, ancilla);
        ancilla = x.getResult();

        mlir::ValueRange control_results = x.getControlResults();
        assert(control_results.size() == controls.size()
               && "wrong number of output control qubits");
        size_t n_fixed_controls = control_qubits.size();
        control_qubits.clear();
        control_qubits.append(control_results.begin(),
                              control_results.begin() + n_fixed_controls);

        for (size_t j = 0; j < vec.getBitWidth(); j++) {
            mlir::Value ctrl = control_results[n_fixed_controls + j];
            bool bit = vec[vec.getBitWidth()-1-j];
            if (!bit) {
                // Undo bit flip above
                ctrl = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                    ctrl).getResult();
            }
            qubits[qubit_idx + j] = ctrl;
        }

        ancillas[ancilla_idx] = ancilla;
    }

    for (mlir::Value ancilla : ancillas) {
        builder.create<qcirc::QfreeZeroOp>(loc, ancilla);
    }
}

} // namespace tweedledum
