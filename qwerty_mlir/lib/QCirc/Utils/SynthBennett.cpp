#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"

// Synthesizes a Bennett embedding, i.e., a circuit U that achieves
// U|x⟩|y⟩ = |x⟩|y⊕f(x)⟩. Assumes that the input circuit is a XAG.
// (TODO: define what that even means)
// This is based on the following paper:
// https://doi.org/10.23919/DATE51398.2021.9474163

namespace {

void synthInPlace(
        mlir::Value val_to_synth,
        size_t tgt_qubit_start_idx,
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        llvm::DenseMap<mlir::Value, size_t> &input_wires);

void synthOutOfPlace(
        mlir::Value val_to_synth,
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        llvm::DenseMap<mlir::Value, size_t> &input_wires,
        std::function<void(size_t)> callback) {
    size_t val_dim = llvm::cast<ccirc::WireType>(
        val_to_synth.getType()).getDim();
    assert(val_dim == 1
           && "Can only synthesize 1Q ops out of place");

    // Easy case where this is an argument: just use the argument qubit
    if (auto input_wire_iter = input_wires.find(val_to_synth);
            input_wire_iter != input_wires.end()) {
        size_t ctrl_qubit_idx = input_wire_iter->second;
        callback(ctrl_qubit_idx);
        // No uncompute needed
    // Next possibility: a NOT. Let's compute the operand somewhere, flip it,
    // use it, and then flip it back again.
    } else if (ccirc::NotOp not_op =
            val_to_synth.getDefiningOp<ccirc::NotOp>()) {
        assert(not_op.getResult().getType().getDim() == 1
               && "Expected every NOT in a XAG to have 1 result");
        synthOutOfPlace(not_op.getOperand(), builder, loc, qubits, input_wires,
            [&](size_t operand_qubit_idx) {
                // Flip qubit for a sec
                qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                    qubits[operand_qubit_idx]);
                qubits[operand_qubit_idx] = x.getResult();

                callback(operand_qubit_idx);

                // Uncompute: flip it back again
                qcirc::Gate1QOp undo_x = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                    qubits[operand_qubit_idx]);
                qubits[operand_qubit_idx] = undo_x.getResult();
            });
    // An AND or parity operation is weird. We will allocate an ancilla qubit
    // and then
    // do in-place parity synth on it, use it, then re-do in-place parity to
    // uncompute it so we can free it.
    } else {
        assert((val_to_synth.getDefiningOp<ccirc::ParityOp>()
                || val_to_synth.getDefiningOp<ccirc::AndOp>())
               && "Purported XAG not in canon XAG form");

        size_t ancilla_idx = qubits.size();
        qubits.push_back(builder.create<qcirc::QallocOp>(loc).getResult());
        synthInPlace(val_to_synth, ancilla_idx, builder, loc, qubits,
                     input_wires);
        callback(ancilla_idx);
        // Uncompute
        synthInPlace(val_to_synth, ancilla_idx, builder, loc, qubits,
                     input_wires);
        builder.create<qcirc::QfreeZeroOp>(loc, qubits.pop_back_val());
    }
}

void synthInPlace(
        mlir::Value val_to_synth,
        size_t tgt_qubit_start_idx,
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &qubits,
        llvm::DenseMap<mlir::Value, size_t> &input_wires) {
    size_t val_dim = llvm::cast<ccirc::WireType>(
        val_to_synth.getType()).getDim();

    // First possibility: this is just an argument. If so, copy with CNOTs
    if (auto input_wire_iter = input_wires.find(val_to_synth);
            input_wire_iter != input_wires.end()) {
        size_t ctrl_qubit_start_idx = input_wire_iter->second;
        for (size_t i = 0; i < val_dim; i++) {
            size_t ctrl_qubit_idx = ctrl_qubit_start_idx + i;
            size_t tgt_qubit_idx = tgt_qubit_start_idx + i;
            qcirc::Gate1QOp cnot = builder.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X,
                std::initializer_list<mlir::Value>{qubits[ctrl_qubit_idx]},
                qubits[tgt_qubit_idx]);
            assert(cnot.getControlResults().size() == 1
                   && "Wrong number of control results for CNOT");
            qubits[ctrl_qubit_idx] = cnot.getControlResults()[0];
            qubits[tgt_qubit_idx] = cnot.getResult();
        }
    // Next possibility: a NOT. If so, we recurse and insert an X gate
    } else if (ccirc::NotOp not_op =
            val_to_synth.getDefiningOp<ccirc::NotOp>()) {
        assert(not_op.getResult().getType().getDim() == 1
               && "Expected every NOT in a XAG to have 1 result");
        synthInPlace(not_op.getOperand(), tgt_qubit_start_idx, builder, loc,
                     qubits, input_wires);
        qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
            loc, qcirc::Gate1Q::X, mlir::ValueRange(),
            qubits[tgt_qubit_start_idx]);
        qubits[tgt_qubit_start_idx] = x.getResult();
    // A parity operation is really easy: just compute everything in-place now
    } else if (ccirc::ParityOp parity =
            val_to_synth.getDefiningOp<ccirc::ParityOp>()) {
        for (mlir::Value operand : parity.getOperands()) {
            synthInPlace(operand, tgt_qubit_start_idx, builder, loc, qubits,
                         input_wires);
        }
    // An AND is a little more annoying. We need to compute both operands
    // out-of-place first.
    } else if (ccirc::AndOp and_op =
            val_to_synth.getDefiningOp<ccirc::AndOp>()) {
        synthOutOfPlace(and_op.getLeft(), builder, loc, qubits, input_wires,
            [&](size_t left_idx) {
                synthOutOfPlace(and_op.getRight(), builder, loc, qubits,
                                input_wires,
                    [&](size_t right_idx) {
                        qcirc::Gate1QOp toffoli =
                            builder.create<qcirc::Gate1QOp>(
                                loc, qcirc::Gate1Q::X,
                                std::initializer_list<mlir::Value>{
                                    qubits[left_idx], qubits[right_idx]},
                                qubits[tgt_qubit_start_idx]);
                        assert(toffoli.getControlResults().size() == 2
                               && "Wrong number of Toffoli control results");
                        qubits[left_idx] = toffoli.getControlResults()[0];
                        qubits[right_idx] = toffoli.getControlResults()[1];
                        qubits[tgt_qubit_start_idx] = toffoli.getResult();
                    });
            });
    } else if (ccirc::ConstantOp const_op =
            val_to_synth.getDefiningOp<ccirc::ConstantOp>()) {
        llvm::APInt val = const_op.getValue();
        assert(val.getBitWidth() == 1
               && "Expected constants in canon XAG to be 1-bit");
        if (val.isOne()) {
            qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                qubits[tgt_qubit_start_idx]);
            qubits[tgt_qubit_start_idx] = x.getResult();
        } else { // val.isZero()
            // Nothing to do
        }
    } else {
        assert(0 && "Purported XAG not in canon XAG form");
    }
}

} // namespace

namespace qcirc {

void synthBennettFromXAG(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        ccirc::CircuitOp xag_circ,
        llvm::SmallVectorImpl<mlir::Value> &init_qubits,
        size_t qubit_idx) {
    size_t in_dim = xag_circ.inDim();
    size_t out_dim = xag_circ.outDim();
    size_t dim = in_dim + out_dim;
    assert(init_qubits.size() - qubit_idx >= dim
           && "Too few qubits for Bennett embedding");

    llvm::SmallVector<mlir::Value> qubits(
        init_qubits.begin() + qubit_idx,
        init_qubits.begin() + qubit_idx + dim);

    // Keep track of which input wires have which bit indices
    llvm::DenseMap<mlir::Value, size_t> input_wires;
    size_t arg_wire_idx = 0;
    for (mlir::BlockArgument arg : xag_circ.bodyBlock().getArguments()) {
        [[maybe_unused]] auto res = input_wires.insert({arg, arg_wire_idx});
        assert(res.second && "Duplicate block arg?");

        for (mlir::Operation *user : arg.getUsers()) {
            if (ccirc::WireUnpackOp unpack =
                    llvm::dyn_cast<ccirc::WireUnpackOp>(user)) {
                size_t this_arg_wire_idx = arg_wire_idx;
                for (mlir::Value unpacked_wire : unpack.getWires()) {
                    [[maybe_unused]] auto this_res = input_wires.insert(
                        {unpacked_wire, this_arg_wire_idx});
                    assert(this_res.second && "Duplicate unpacked arg?");
                    this_arg_wire_idx++;
                }
            }
        }

        size_t arg_dim = llvm::cast<ccirc::WireType>(arg.getType()).getDim();
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
                synthInPlace(pack_arg, ret_qubit_idx, builder, loc, qubits,
                             input_wires);
                size_t pack_arg_dim = llvm::cast<ccirc::WireType>(
                    pack_arg.getType()).getDim();
                ret_qubit_idx += pack_arg_dim;
            }
        } else {
            synthInPlace(ret_val, ret_qubit_idx, builder, loc, qubits,
                         input_wires);
            ret_qubit_idx += ret_dim;
        }
    }

    assert(qubits.size() == dim
           && "Wrong number of qubits. Ancillas not freed yet?");
    for (size_t i = 0; i < dim; i++) {
        init_qubits[qubit_idx + i] = qubits[i];
    }
}

} // namespace qcirc
