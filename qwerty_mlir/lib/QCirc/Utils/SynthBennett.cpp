#include <variant>
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"

// Synthesizes a Bennett embedding, i.e., a circuit U that achieves
// U|x⟩|y⟩ = |x⟩|y⊕f(x)⟩. Assumes that the input circuit is a XAG.
// (TODO: define what that even means)
// This is based on the following paper:
// https://doi.org/10.23919/DATE51398.2021.9474163

namespace {

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
};

struct ParityQubits {
    bool complemented;
    llvm::SmallVector<WireQubit> qubits;
};

struct Synthesizer {
    mlir::OpBuilder &builder;
    mlir::Location loc;
    llvm::SmallVector<mlir::Value> qubits;
    llvm::DenseMap<mlir::Value, WireQubit> wire_qubits;

    Synthesizer(mlir::OpBuilder &builder,
                mlir::Location loc,
                llvm::SmallVectorImpl<mlir::Value> &init_qubits)
               : builder(builder), loc(loc),
                 qubits(init_qubits.begin(), init_qubits.end()) {}

    void xorWireInto(mlir::Value wire, size_t tgt_qubit_start_idx) {
        size_t wire_dim = llvm::cast<ccirc::WireType>(
            wire.getType()).getDim();

        // First possibility: this is just an argument. If so, copy with CNOTs
        if (auto input_wire_iter = wire_qubits.find(wire);
                input_wire_iter != wire_qubits.end()) {
            Wire &wire = input_wire_iter->second;
            assert(wire.holds_alternative<InputQubit>()
                   && "Cannot XOR into a node that already exists");
            size_t ctrl_qubit_start_idx = std::get<InputQubit>(wire).qubit_idx;

            for (size_t i = 0; i < wire_dim; i++) {
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
                wire.getDefiningOp<ccirc::NotOp>()) {
            assert(not_op.getResult().getType().getDim() == 1
                   && "Expected every NOT in a XAG to have 1 result");
            xorWireInto(not_op.gerOperand(), tgt_qubit_start_idx);
            qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
                loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                qubits[tgt_qubit_start_idx]);
            qubits[tgt_qubit_start_idx] = x.getResult();
        // A parity operation is really easy: just compute everything in-place now
        } else if (ccirc::ParityOp parity =
                wire.getDefiningOp<ccirc::ParityOp>()) {
            for (mlir::Value operand : parity.getOperands()) {
                xorNodeInto(operand, tgt_qubit_start_idx);
            }
        // An AND is a little more annoying. We need to compute both operands
        // out-of-place first.
        } else if (ccirc::AndOp and_op =
                wire.getDefiningOp<ccirc::AndOp>()) {
            mlir::Value left = and_op.getLeft();
            mlir::Value right = and_op.getRight();
            //size_t left_idx = getWireAsQubit(left, builder, loc, qubits, wires);
            //size_t right_idx = getWireAsQubit(right, builder, loc, qubits, wires);
            // TODO: getWireAsParityQubits for both left and right
            // TODO: do Bruno's parity trick
            // TODO: if complement, add X gate

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

            // TODO: if complement, undo X gate
            // TODO: undo Bruno's parity trick
            // TODO: pop undos off "stack" (really, list of wires) one-by-one,
            //       decrementing refcount and undoing them if refcount == 0
        } else if (ccirc::ConstantOp const_op =
                wire.getDefiningOp<ccirc::ConstantOp>()) {
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
                    xorNodeInto(pack_arg, ret_qubit_idx, builder, loc, qubits,
                                wires);
                    size_t pack_arg_dim = llvm::cast<ccirc::WireType>(
                        pack_arg.getType()).getDim();
                    ret_qubit_idx += pack_arg_dim;
                }
            } else {
                xorNodeInto(ret_val, ret_qubit_idx, builder, loc, qubits,
                            wires);
                ret_qubit_idx += ret_dim;
            }
        }
    }

    void finish(llvm::SmallVectorImpl<mlir::Value> &qubits_out) {
        assert(qubits_out.size() == qubits.size()
               && "Wrong number of qubits. Missing free()ing ancilla?");
        qubits_out.clear();
        qubits_out.append(qubits.begin(), qubits.end());
    }
};

} // namespace

namespace qcirc {

void synthBennettFromXAG(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        ccirc::CircuitOp xag_circ,
        llvm::SmallVectorImpl<mlir::Value> &init_qubits) {
    Synthesizer synth(builder, loc, qubits);
    synth.synthesize(xag_circ);
    synth.finish(init_qubits);
} // namespace qcirc

//struct InputQubit {
//    size_t qubit_idx;
//
//    InputQubit(size_t qubit_idx) : qubit_idx(qubit_idx) {}
//};
//
//struct AndAncilla {
//    size_t qubit_idx;
//    size_t refcount;
//
//    // TODO: also contains the Value, right? for undoing?
//
//    AndAncilla(size_t qubit_idx, size_t refcount)
//              : qubit_idx(qubit_idx),
//                uses_left(refcount) {}
//};
//
//using Wire = std::variant<InputQubit, AndAncilla>;
//using Wires = llvm::DenseMap<mlir::Value, Wire>;
//
//void xorNodeInto(
//        mlir::Value val_to_synth,
//        size_t tgt_qubit_start_idx,
//        mlir::OpBuilder &builder,
//        mlir::Location loc,
//        llvm::SmallVectorImpl<mlir::Value> &qubits,
//        Wires &wires);
//
//// TODO: synth_and()
////     TODO: look up val in wires
////       TODO: if exists, return just that qubit index
////       TODO: if not, allocate an ancilla and call xorNodeInto(). return that ancilla index
//
//(List<Wire>, bool) getWireAsParityQubits(
//        mlir::Value val_to_synth,
//        mlir::OpBuilder &builder,
//        mlir::Location loc,
//        llvm::SmallVectorImpl<mlir::Value> &qubits,
//        Wires &wires) {
//    size_t val_dim = llvm::cast<ccirc::WireType>(
//        val_to_synth.getType()).getDim();
//    assert(val_dim == 1
//           && "Can only synthesize a 1-bit wire to a single qubit");
//
//    // Already synthesized (or it's an input)
//    if (auto wire_iter = wires.find(val_to_synth);
//            wire_iter != wires.end()) {
//        if (wire_iter->holds_alternative<InputQubit>()) {
//            return std::get<InputQubit>(*wire_iter).qubit_idx;
//        } else if (wire_iter->holds_alternative<AndAncilla>()) {
//            AndAncilla &synth = std::get<AndAncilla>(*wire_iter);
//            //synth.uses_left--;
//            return synth.qubit_idx;
//        }
//    } else {
//        // TODO: pop off a not if present
//        // TODO: if next op is an AND, call synth_and()
//        // TODO: if next op is a parity,
//        //       TODO: call synth_and() for each operand
//        //       TODO: return list of all results
//    }
//}
//
//void xorNodeInto(
//        mlir::Value val_to_synth,
//        size_t tgt_qubit_start_idx,
//        mlir::OpBuilder &builder,
//        mlir::Location loc,
//        llvm::SmallVectorImpl<mlir::Value> &qubits,
//        Wires &wires) {
//    size_t val_dim = llvm::cast<ccirc::WireType>(
//        val_to_synth.getType()).getDim();
//
//    // First possibility: this is just an argument. If so, copy with CNOTs
//    if (auto input_wire_iter = wires.find(val_to_synth);
//            input_wire_iter != wires.end()) {
//        Wire &wire = input_wire_iter->second;
//        assert(wire.holds_alternative<InputQubit>()
//               && "Cannot XOR into a node that already exists");
//        size_t ctrl_qubit_start_idx = std::get<InputQubit>(wire).qubit_idx;
//
//        for (size_t i = 0; i < val_dim; i++) {
//            size_t ctrl_qubit_idx = ctrl_qubit_start_idx + i;
//            size_t tgt_qubit_idx = tgt_qubit_start_idx + i;
//            qcirc::Gate1QOp cnot = builder.create<qcirc::Gate1QOp>(
//                loc, qcirc::Gate1Q::X,
//                std::initializer_list<mlir::Value>{qubits[ctrl_qubit_idx]},
//                qubits[tgt_qubit_idx]);
//            assert(cnot.getControlResults().size() == 1
//                   && "Wrong number of control results for CNOT");
//            qubits[ctrl_qubit_idx] = cnot.getControlResults()[0];
//            qubits[tgt_qubit_idx] = cnot.getResult();
//        }
//    // Next possibility: a NOT. If so, we recurse and insert an X gate
//    } else if (ccirc::NotOp not_op =
//            val_to_synth.getDefiningOp<ccirc::NotOp>()) {
//        assert(not_op.getResult().getType().getDim() == 1
//               && "Expected every NOT in a XAG to have 1 result");
//        xorNodeInto(not_op.getOperand(), tgt_qubit_start_idx, builder, loc,
//                    qubits, wires);
//        qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
//            loc, qcirc::Gate1Q::X, mlir::ValueRange(),
//            qubits[tgt_qubit_start_idx]);
//        qubits[tgt_qubit_start_idx] = x.getResult();
//    // A parity operation is really easy: just compute everything in-place now
//    } else if (ccirc::ParityOp parity =
//            val_to_synth.getDefiningOp<ccirc::ParityOp>()) {
//        for (mlir::Value operand : parity.getOperands()) {
//            xorNodeInto(operand, tgt_qubit_start_idx, builder, loc, qubits,
//                        wires);
//        }
//    // An AND is a little more annoying. We need to compute both operands
//    // out-of-place first.
//    } else if (ccirc::AndOp and_op =
//            val_to_synth.getDefiningOp<ccirc::AndOp>()) {
//        mlir::Value left = and_op.getLeft();
//        mlir::Value right = and_op.getRight();
//        //size_t left_idx = getWireAsQubit(left, builder, loc, qubits, wires);
//        //size_t right_idx = getWireAsQubit(right, builder, loc, qubits, wires);
//        // TODO: getWireAsParityQubits for both left and right
//        // TODO: do Bruno's parity trick
//        // TODO: if complement, add X gate
//
//        qcirc::Gate1QOp toffoli =
//            builder.create<qcirc::Gate1QOp>(
//                loc, qcirc::Gate1Q::X,
//                std::initializer_list<mlir::Value>{
//                    qubits[left_idx], qubits[right_idx]},
//                qubits[tgt_qubit_start_idx]);
//        assert(toffoli.getControlResults().size() == 2
//               && "Wrong number of Toffoli control results");
//        qubits[left_idx] = toffoli.getControlResults()[0];
//        qubits[right_idx] = toffoli.getControlResults()[1];
//        qubits[tgt_qubit_start_idx] = toffoli.getResult();
//
//        // TODO: if complement, undo X gate
//        // TODO: undo Bruno's parity trick
//        // TODO: pop undos off "stack" (really, list of wires) one-by-one,
//        //       decrementing refcount and undoing them if refcount == 0
//    } else if (ccirc::ConstantOp const_op =
//            val_to_synth.getDefiningOp<ccirc::ConstantOp>()) {
//        llvm::APInt val = const_op.getValue();
//        assert(val.getBitWidth() == 1
//               && "Expected constants in canon XAG to be 1-bit");
//        if (val.isOne()) {
//            qcirc::Gate1QOp x = builder.create<qcirc::Gate1QOp>(
//                loc, qcirc::Gate1Q::X, mlir::ValueRange(),
//                qubits[tgt_qubit_start_idx]);
//            qubits[tgt_qubit_start_idx] = x.getResult();
//        } else { // val.isZero()
//            // Nothing to do
//        }
//    } else {
//        assert(0 && "Purported XAG not in canon XAG form");
//    }
//}
//

//    size_t in_dim = xag_circ.inDim();
//    size_t out_dim = xag_circ.outDim();
//    size_t dim = in_dim + out_dim;
//    assert(init_qubits.size() - qubit_idx >= dim
//           && "Too few qubits for Bennett embedding");
//
//    llvm::SmallVector<mlir::Value> qubits(
//        init_qubits.begin() + qubit_idx,
//        init_qubits.begin() + qubit_idx + dim);
//
//    // Keep track of which input wires have which bit indices
//    Wires wires;
//    size_t arg_wire_idx = 0;
//    for (mlir::BlockArgument arg : xag_circ.bodyBlock().getArguments()) {
//        [[maybe_unused]] auto res = wires.insert({arg, Wire(InputQubit(arg_wire_idx))});
//        assert(res.second && "Duplicate block arg?");
//
//        for (mlir::Operation *user : arg.getUsers()) {
//            if (ccirc::WireUnpackOp unpack =
//                    llvm::dyn_cast<ccirc::WireUnpackOp>(user)) {
//                size_t this_arg_wire_idx = arg_wire_idx;
//                for (mlir::Value unpacked_wire : unpack.getWires()) {
//                    [[maybe_unused]] auto this_res = wires.insert(
//                        {unpacked_wire, Wire(InputQubit(this_arg_wire_idx))});
//                    assert(this_res.second && "Duplicate unpacked arg?");
//                    this_arg_wire_idx++;
//                }
//            }
//        }
//
//        size_t arg_dim = llvm::cast<ccirc::WireType>(arg.getType()).getDim();
//        arg_wire_idx += arg_dim;
//    }
//
//    ccirc::ReturnOp ret = llvm::cast<ccirc::ReturnOp>(
//        xag_circ.bodyBlock().getTerminator());
//    size_t ret_qubit_idx = in_dim;
//    for (mlir::Value ret_val : ret.getOperands()) {
//        size_t ret_dim = llvm::cast<ccirc::WireType>(
//            ret_val.getType()).getDim();
//        assert(ret_dim && "Zero dimension for wire");
//
//        if (ccirc::WirePackOp pack =
//                ret_val.getDefiningOp<ccirc::WirePackOp>()) {
//            for (mlir::Value pack_arg : pack.getWires()) {
//                xorNodeInto(pack_arg, ret_qubit_idx, builder, loc, qubits,
//                            wires);
//                size_t pack_arg_dim = llvm::cast<ccirc::WireType>(
//                    pack_arg.getType()).getDim();
//                ret_qubit_idx += pack_arg_dim;
//            }
//        } else {
//            xorNodeInto(ret_val, ret_qubit_idx, builder, loc, qubits,
//                        wires);
//            ret_qubit_idx += ret_dim;
//        }
//    }
//
//    assert(qubits.size() == dim
//           && "Wrong number of qubits. Ancillas not freed yet?");
//    for (size_t i = 0; i < dim; i++) {
//        init_qubits[qubit_idx + i] = qubits[i];
//    }
//}
//} // namespace qcirc
