#include "util.hpp"
#include <algorithm>
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Synth/QCircSynth.h"

// Synthesizes a Bennett embedding, i.e., a circuit U that achieves
// U|x⟩|y⟩ = |x⟩|y⊕f(x)⟩. Assumes that the input circuit is a XAG.
// (TODO: define what that even means)
// This is based on the following paper:
// https://doi.org/10.23919/DATE51398.2021.9474163

namespace {

enum SynthFlags {
    // Intermediate computation (targeting an ancilla) versus output
    // computation (targeting an output qubit). Used to determine if we can
    // safely use Selinger's Toffoli trick to reduce T counts.
    FLAG_TMP = 0 << 0,
    FLAG_OUT = 1 << 0,
    // Forward and reverse computation, respectively.
    FLAG_FWD = 0 << 1,
    FLAG_REV = 1 << 1,
};

struct WireQubit {
    enum class Kind {
        InputQubit,
        AndAncilla,
    };

    mlir::Value wire;
    size_t qubit_idx;
    size_t refcount;
    Kind kind;

    WireQubit(mlir::Value wire, size_t qubit_idx)
             : wire(wire), qubit_idx(qubit_idx), kind(Kind::InputQubit) {}

    WireQubit(mlir::Value wire, size_t qubit_idx, size_t refcount)
             : wire(wire), qubit_idx(qubit_idx), refcount(refcount),
               kind(Kind::AndAncilla) {}

    bool wasFreed() {
        return kind == Kind::AndAncilla && !refcount;
    }
};

struct ParityQubits {
    bool complemented;
    llvm::SmallVector<WireQubit> qubits;

    ParityQubits() : complemented(false) {}

    void join(ParityQubits &&other) {
        complemented ^= other.complemented;
        qubits.append(other.qubits);
    }

    void sortedQubitIndices(llvm::SmallVectorImpl<size_t> &indices_out) {
        indices_out.clear();
        for (WireQubit &qubit : qubits) {
            indices_out.push_back(qubit.qubit_idx);
        }
        std::sort(indices_out.begin(), indices_out.end());
    }
};

struct Synthesizer {
    mlir::OpBuilder &builder;
    mlir::Location loc;
    llvm::SmallVector<mlir::Value> qubits;
    llvm::DenseMap<mlir::Value, WireQubit> wire_qubits;
    // Cached quantum.calc ops for pi and -pi
    mlir::Value pi, neg_pi;

    Synthesizer(mlir::OpBuilder &builder,
                mlir::Location loc,
                llvm::SmallVectorImpl<mlir::Value> &init_qubits)
               : builder(builder), loc(loc),
                 qubits(init_qubits.begin(), init_qubits.end()) {}

    void freeAncillaForAnd(WireQubit &to_free) {
        // We need to modify the WireQubit stored in the map, not the argument
        // on the stack.
        auto wire_it = wire_qubits.find(to_free.wire);
        assert(wire_it != wire_qubits.end()
               && "Freeing qubit that is not allocated");
        WireQubit &qubit = wire_it->getSecond();

        if (qubit.kind == WireQubit::Kind::AndAncilla) {
            if (!--qubit.refcount) {
                // Undo computation on this ancilla now that we're done with it
                xorWireInto(qubit.wire, qubit.qubit_idx,
                            static_cast<SynthFlags>(FLAG_TMP | FLAG_REV));
                builder.create<qcirc::QfreeZeroOp>(
                    loc, qubits[qubit.qubit_idx]);
                // We don't delete this from the array because we may be
                // freeing qubits out of order. But we can at least set it to a
                // null pointer to cause a nuclear explosion if some code
                // incorrectly tries to use it.
                qubits[qubit.qubit_idx] = nullptr;

                // Also need to decrement the refcount for any dependencies of
                // this AND that we were keeping around
                ccirc::AndOp and_op = qubit.wire.getDefiningOp<ccirc::AndOp>();
                assert(and_op && "AND ancilla not defined for AND, how?");
                propagateFreeUpward(and_op.getLeft());
                propagateFreeUpward(and_op.getRight());
            }
        } else {
            // Input qubit, no freeing needed
        }
    }

    void propagateFreeUpward(mlir::Value value) {
        if (auto wire_iter = wire_qubits.find(value);
                wire_iter != wire_qubits.end()) {
            freeAncillaForAnd(wire_iter->getSecond());
        // This could be a ccirc.constant, ccirc.parity, or ccirc.not. The code
        // below will do nothing for the first and propagate the free() upward
        // for the other two cases.
        } else {
            mlir::Operation *op = value.getDefiningOp();
            for (mlir::Value operand : op->getOperands()) {
                propagateFreeUpward(operand);
            }
        }
    }

    WireQubit synthAndIfNeeded(ccirc::AndOp and_op) {
        mlir::Value wire = and_op.getResult();

        // Already synthesized
        if (auto wire_iter = wire_qubits.find(wire);
                wire_iter != wire_qubits.end()) {
            return wire_iter->getSecond();
        // Not yet synthesized. Synthesize it!
        } else {
            size_t ancilla_idx = qubits.size();
            mlir::Value ancilla =
                builder.create<qcirc::QallocOp>(loc).getResult();
            qubits.push_back(ancilla);

            xorWireInto(wire, ancilla_idx,
                        static_cast<SynthFlags>(FLAG_TMP | FLAG_FWD));

            // By definition of XAG, every user will use each of its operands
            // exactly once. Thus, this is equivalent to the number of users.
            WireQubit qubit(wire, ancilla_idx, wire.getNumUses());
            wire_qubits.insert({wire, qubit});
            return qubit;
        }
    }

    ParityQubits parityQubitsForWire(mlir::Value wire) {
        ParityQubits parity;

        size_t val_dim = llvm::cast<ccirc::WireType>(
            wire.getType()).getDim();
        assert(val_dim == 1
               && "Can only synthesize a 1-bit wire to a single qubit");

        // Already synthesized (or it's an input)
        if (auto wire_iter = wire_qubits.find(wire);
                wire_iter != wire_qubits.end()) {
            parity.qubits.push_back(wire_iter->getSecond());
        // Not yet synthesized
        } else {
            // Strip off initial NOT op
            if (ccirc::NotOp not_op = wire.getDefiningOp<ccirc::NotOp>()) {
                parity.complemented = true;
                wire = not_op.getOperand();
            }

            if (ccirc::AndOp and_op = wire.getDefiningOp<ccirc::AndOp>()) {
                parity.qubits.push_back(synthAndIfNeeded(and_op));
            } else if (ccirc::ParityOp parity_op =
                    wire.getDefiningOp<ccirc::ParityOp>()) {
                for (mlir::Value operand : parity_op.getOperands()) {
                    parity.join(parityQubitsForWire(operand));
                }
            } else {
                assert(0 && "Purported XAG not in canon XAG form");
            }
        }

        return parity;
    }

    void runXGate(size_t qubit_idx) {
        qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[qubit_idx]);
        assert(x.getControlResults().empty()
               && "Expected no control results for X gate");
        qubits[qubit_idx] = x.getResult();
    }

    void runCNOTGate(size_t ctrl_idx, size_t tgt_idx) {
        qcirc::Gate1QOp cnot = builder.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::X,
            std::initializer_list<mlir::Value>{qubits[ctrl_idx]},
            qubits[tgt_idx]);
        assert(cnot.getControlResults().size() == 1
               && "Wrong number of control results for CNOT");
        qubits[ctrl_idx] = cnot.getControlResults()[0];
        qubits[tgt_idx] = cnot.getResult();
    }

    void runToffoliGate(size_t ctrl1_idx, size_t ctrl2_idx, size_t tgt_idx) {
        qcirc::Gate1QOp toffoli =
            builder.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X,
                std::initializer_list<mlir::Value>{
                    qubits[ctrl1_idx], qubits[ctrl2_idx]},
                qubits[tgt_idx]);
        assert(toffoli.getControlResults().size() == 2
               && "Wrong number of Toffoli control results");
        qubits[ctrl1_idx] = toffoli.getControlResults()[0];
        qubits[ctrl2_idx] = toffoli.getControlResults()[1];
        qubits[tgt_idx] = toffoli.getResult();
    }

    inline mlir::Value getPi() {
        if (!pi) {
            pi = qcirc::stationaryF64Const(builder, loc, M_PI);
        }
        return pi;
    }

    inline mlir::Value getNegPi() {
        if (!neg_pi) {
            neg_pi = qcirc::stationaryF64Const(builder, loc, -M_PI);
        }
        return neg_pi;
    }

    // This is due to Selinger (Equations 10 and 11):
    // https://doi.org/10.1103/PhysRevA.87.042302
    // We follow Tweedledum's lead and indicate this by creating an Rx(±π) with
    // two controls.
    void runSelingerToffoliGate(size_t ctrl1_idx, size_t ctrl2_idx,
                                size_t tgt_idx, bool rev) {
        qcirc::Gate1Q1POp ccrx =
            builder.create<qcirc::Gate1Q1POp>(
                loc, qcirc::Gate1Q1P::Rx,
                rev? getNegPi() : getPi(),
                std::initializer_list<mlir::Value>{
                    qubits[ctrl1_idx], qubits[ctrl2_idx]},
                qubits[tgt_idx]);
        assert(ccrx.getControlResults().size() == 2
               && "Wrong number of CCRx control results");
        qubits[ctrl1_idx] = ccrx.getControlResults()[0];
        qubits[ctrl2_idx] = ccrx.getControlResults()[1];
        qubits[tgt_idx] = ccrx.getResult();
    }

    template <typename It>
    void parityInPlace(size_t tgt_idx,
                       It begin_ctrl,
                       It end_ctrl) {
        for (auto it = begin_ctrl; it != end_ctrl; it++) {
            runCNOTGate(*it, tgt_idx);
        }
    }

    void xorAndWireInto(ccirc::AndOp and_op, size_t tgt_qubit_idx,
                        SynthFlags flags) {
        mlir::Value left_wire = and_op.getLeft();
        mlir::Value right_wire = and_op.getRight();

        ParityQubits left_parity = parityQubitsForWire(left_wire);
        ParityQubits right_parity = parityQubitsForWire(right_wire);

        llvm::SmallVector<size_t> left_qubit_indices, right_qubit_indices;
        left_parity.sortedQubitIndices(left_qubit_indices);
        right_parity.sortedQubitIndices(right_qubit_indices);

        llvm::SmallVector<size_t> shared_indices;
        std::set_intersection(left_qubit_indices.begin(),
                              left_qubit_indices.end(),
                              right_qubit_indices.begin(),
                              right_qubit_indices.end(),
                              std::back_inserter(shared_indices));
        llvm::SmallVector<size_t> only_left_indices;
        std::set_difference(left_qubit_indices.begin(),
                            left_qubit_indices.end(),
                            right_qubit_indices.begin(),
                            right_qubit_indices.end(),
                            std::back_inserter(only_left_indices));
        llvm::SmallVector<size_t> only_right_indices;
        std::set_difference(right_qubit_indices.begin(),
                            right_qubit_indices.end(),
                            left_qubit_indices.begin(),
                            left_qubit_indices.end(),
                            std::back_inserter(only_right_indices));

        size_t left_ctrl_idx, right_ctrl_idx;
        // Easy case: compute in-place parity on each set of operands
        if (shared_indices.empty()) {
            parityInPlace(only_left_indices[0],
                          only_left_indices.begin()+1,
                          only_left_indices.end());
            parityInPlace(only_right_indices[0],
                          only_right_indices.begin()+1,
                          only_right_indices.end());
            left_ctrl_idx = only_left_indices[0];
            right_ctrl_idx = only_right_indices[0];
        // Use Bruno Schmitt's trick for the trickier case: compute the
        // parity of the shared qubits in-place
        } else {
            parityInPlace(shared_indices[0],
                          shared_indices.begin()+1, shared_indices.end());

            if (!only_left_indices.empty() && !only_right_indices.empty()) {
                parityInPlace(only_left_indices[0],
                              only_left_indices.begin()+1,
                              only_left_indices.end());
                runCNOTGate(shared_indices[0], only_left_indices[0]);
                parityInPlace(shared_indices[0],
                              only_right_indices.begin(),
                              only_right_indices.end());

                left_ctrl_idx = only_left_indices[0];
                right_ctrl_idx = shared_indices[0];
            } else if (only_left_indices.empty()
                       && !only_right_indices.empty()) {
                parityInPlace(only_right_indices[0],
                              only_right_indices.begin()+1,
                              only_right_indices.end());
                runCNOTGate(shared_indices[0], only_right_indices[0]);

                left_ctrl_idx = shared_indices[0];
                right_ctrl_idx = only_right_indices[0];
            } else if (!only_left_indices.empty()
                       && only_right_indices.empty()) {
                parityInPlace(only_left_indices[0],
                              only_left_indices.begin()+1,
                              only_left_indices.end());
                runCNOTGate(shared_indices[0], only_left_indices[0]);

                left_ctrl_idx = only_left_indices[0];
                right_ctrl_idx = shared_indices[0];
            } else {
                // TODO: we can support this by just XORing all the shared
                //       indices into the target qubit (maybe with an
                //       additional bit flip if complemented)
                assert(0 && "Both operands of an AND should not be the same");
            }
        }

        if (left_parity.complemented) {
            runXGate(left_ctrl_idx);
        }
        if (right_parity.complemented) {
            runXGate(right_ctrl_idx);
        }

        if ((flags & FLAG_OUT)) {
            runToffoliGate(left_ctrl_idx, right_ctrl_idx, tgt_qubit_idx);
        } else { // FLAG_TMP
            bool is_reverse = !!(flags & FLAG_REV);
            runSelingerToffoliGate(left_ctrl_idx, right_ctrl_idx,
                                   tgt_qubit_idx, is_reverse);
        }

        // Undo complements
        if (right_parity.complemented) {
            runXGate(right_ctrl_idx);
        }
        if (left_parity.complemented) {
            runXGate(left_ctrl_idx);
        }

        // Undo Bruno Schmitt's in-place parity trick. This is the same
        // code as above, just backwards.
        if (shared_indices.empty()) {
            parityInPlace(only_right_indices[0],
                          only_right_indices.rbegin(),
                          only_right_indices.rend()-1);
            parityInPlace(only_left_indices[0],
                          only_left_indices.rbegin(),
                          only_left_indices.rend()-1);
        } else {
            if (!only_left_indices.empty() && !only_right_indices.empty()) {
                parityInPlace(shared_indices[0],
                              only_right_indices.rbegin(),
                              only_right_indices.rend());
                runCNOTGate(shared_indices[0], only_left_indices[0]);
                parityInPlace(only_left_indices[0],
                              only_left_indices.rbegin(),
                              only_left_indices.rend()-1);
            } else if (only_left_indices.empty()
                       && !only_right_indices.empty()) {
                runCNOTGate(shared_indices[0], only_right_indices[0]);
                parityInPlace(only_right_indices[0],
                              only_right_indices.rbegin(),
                              only_right_indices.rend()-1);
            } else if (!only_left_indices.empty()
                       && only_right_indices.empty()) {
                runCNOTGate(shared_indices[0], only_left_indices[0]);
                parityInPlace(only_left_indices[0],
                              only_left_indices.rbegin(),
                              only_left_indices.rend()-1);
            } else {
                // TODO: we can support this by just XORing all the shared
                //       indices into the target qubit (maybe with an
                //       additional bit flip if complemented)
                assert(0 && "Both operands of an AND should not be the same");
            }

            parityInPlace(shared_indices[0],
                          shared_indices.rbegin(), shared_indices.rend()-1);
        }

        // Just an aesthetic tweak: uncompute in the opposite order that we
        // compute. This makes the circuit symmetrical.
        for (WireQubit &qubit : right_parity.qubits) {
            freeAncillaForAnd(qubit);
        }
        for (WireQubit &qubit : left_parity.qubits) {
            freeAncillaForAnd(qubit);
        }
    }

    void xorWireInto(mlir::Value wire, size_t tgt_qubit_start_idx,
                     SynthFlags flags) {
        size_t wire_dim = llvm::cast<ccirc::WireType>(
            wire.getType()).getDim();

        // First possibility: this is just an argument. If so, copy with CNOTs
        if (auto input_wire_iter = wire_qubits.find(wire);
                input_wire_iter != wire_qubits.end()
                // If we are uncomputing, don't try to do a copy
                && !input_wire_iter->getSecond().wasFreed()) {
            WireQubit &wire = input_wire_iter->getSecond();
            size_t ctrl_qubit_start_idx = wire.qubit_idx;

            for (size_t i = 0; i < wire_dim; i++) {
                size_t ctrl_qubit_idx = ctrl_qubit_start_idx + i;
                size_t tgt_qubit_idx = tgt_qubit_start_idx + i;
                runCNOTGate(ctrl_qubit_idx, tgt_qubit_idx);
            }
        // Next possibility: a NOT. If so, we recurse and insert an X gate
        } else if (ccirc::NotOp not_op =
                wire.getDefiningOp<ccirc::NotOp>()) {
            assert(not_op.getResult().getType().getDim() == 1
                   && "Expected every NOT in a XAG to have 1 result");
            xorWireInto(not_op.getOperand(), tgt_qubit_start_idx, flags);
            runXGate(tgt_qubit_start_idx);
        // A parity operation is really easy: just compute everything in-place
        } else if (ccirc::ParityOp parity =
                wire.getDefiningOp<ccirc::ParityOp>()) {
            for (mlir::Value operand : parity.getOperands()) {
                xorWireInto(operand, tgt_qubit_start_idx, flags);
            }
        // An AND is complicated enough where it deserves its own function,
        // xorAndWireInto()`, but there is still one tricky case to handle
        // here: what if this output wire is also used as an input to other
        // logic (or perhaps another output)? If so, we should allocate an
        // ancilla to store the AND result so that we can copy it to this
        // output wire and then use it wherever else it is needed.
        } else if (ccirc::AndOp and_op =
                wire.getDefiningOp<ccirc::AndOp>()) {
            if ((flags & FLAG_OUT) && wire.getNumUses() > 1) {
                WireQubit qubit = synthAndIfNeeded(and_op);
                runCNOTGate(qubit.qubit_idx, tgt_qubit_start_idx);
                freeAncillaForAnd(qubit);
            } else {
                xorAndWireInto(and_op, tgt_qubit_start_idx, flags);
            }
        } else if (ccirc::ConstantOp const_op =
                wire.getDefiningOp<ccirc::ConstantOp>()) {
            llvm::APInt val = const_op.getValue();
            assert(val.getBitWidth() == 1
                   && "Expected constants in canon XAG to be 1-bit");
            if (val.isOne()) {
                runXGate(tgt_qubit_start_idx);
            } else { // val.isZero()
                // Nothing to do
            }
        } else {
            assert(0 && "Purported XAG not in canon XAG form");
        }
    }

    void synthesize(ccirc::CircuitOp xag_circ) {
        size_t in_dim = xag_circ.inDim();
        size_t out_dim = xag_circ.outDim();
        size_t dim = in_dim + out_dim;
        assert(qubits.size() == dim
               && "Wrong number of qubits for Bennett embedding");

        size_t arg_wire_idx = 0;
        for (mlir::BlockArgument arg : xag_circ.bodyBlock().getArguments()) {
            [[maybe_unused]] auto res =
                wire_qubits.insert({arg, WireQubit(arg, arg_wire_idx)});
            assert(res.second && "Duplicate block arg?");

            // By definition of XAG, every user will use each of its operands
            // exactly once
            for (mlir::Operation *user : arg.getUsers()) {
                if (ccirc::WireUnpackOp unpack =
                        llvm::dyn_cast<ccirc::WireUnpackOp>(user)) {
                    size_t this_arg_wire_idx = arg_wire_idx;
                    for (mlir::Value unpacked_wire : unpack.getWires()) {
                        [[maybe_unused]] auto this_res = wire_qubits.insert(
                            {unpacked_wire,
                             WireQubit(unpacked_wire, this_arg_wire_idx)});
                        assert(this_res.second && "Duplicate unpacked arg?");
                        this_arg_wire_idx++;
                    }
                }
            }

            size_t arg_dim = llvm::cast<ccirc::WireType>(
                arg.getType()).getDim();
            arg_wire_idx += arg_dim;
        }

        ccirc::ReturnOp ret = llvm::cast<ccirc::ReturnOp>(
            xag_circ.bodyBlock().getTerminator());
        size_t ret_qubit_idx = in_dim;
        for (mlir::Value ret_val : ret.getOperands()) {
            size_t ret_dim = llvm::cast<ccirc::WireType>(
                ret_val.getType()).getDim();
            assert(ret_dim && "Zero dimension for wire");

            if (ccirc::WirePackOp pack =
                    ret_val.getDefiningOp<ccirc::WirePackOp>()) {
                for (mlir::Value pack_arg : pack.getWires()) {
                    xorWireInto(pack_arg, ret_qubit_idx,
                                static_cast<SynthFlags>(FLAG_OUT | FLAG_FWD));
                    size_t pack_arg_dim = llvm::cast<ccirc::WireType>(
                        pack_arg.getType()).getDim();
                    ret_qubit_idx += pack_arg_dim;
                }
            } else {
                xorWireInto(ret_val, ret_qubit_idx,
                            static_cast<SynthFlags>(FLAG_OUT | FLAG_FWD));
                ret_qubit_idx += ret_dim;
            }
        }
    }

    void finish(llvm::SmallVectorImpl<mlir::Value> &qubits_out) {
        size_t n_qubits_expected = qubits_out.size();
        assert(n_qubits_expected <= qubits.size() && "Too few qubits");
        qubits_out.clear();
        qubits_out.append(qubits.begin(), qubits.begin() + n_qubits_expected);

#ifndef NDEBUG
        for (size_t i = n_qubits_expected; i < qubits.size(); i++) {
            // When we free ancilla above, we set their entries in this array
            // to null pointers. Thus, this check verifies that we actually
            // freed all ancilla.
            assert(!qubits[i] && "Ancilla was not freed");
        }
#endif
    }
};

} // namespace

namespace qcirc {

void synthBennettFromXAG(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        ccirc::CircuitOp xag_circ,
        llvm::SmallVectorImpl<mlir::Value> &init_qubits) {
    Synthesizer synth(builder, loc, init_qubits);
    synth.synthesize(xag_circ);
    synth.finish(init_qubits);
}

} // namespace qcirc
