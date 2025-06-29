#include <queue>
#include <unordered_map>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/Utils/QCircUtils.h"

// Code for generating OpenQASM 3.0 from QCirc IR

namespace {

// Useful because we can't print out assembly as we parse since we need to
// print out he quantum and classical register declarations first
struct QasmInst {
    std::string name;
    std::vector<double> params;
    std::vector<unsigned int> ctrl_indices;
    std::vector<unsigned int> qubit_indices;
    std::vector<unsigned int> bit_indices;
    mlir::Location loc;

    QasmInst(std::string name,
             std::vector<double> params,
             std::vector<unsigned int> ctrl_indices,
             std::vector<unsigned int> qubit_indices,
             std::vector<unsigned int> bit_indices,
             mlir::Location loc)
            : name(name),
              params(params),
              ctrl_indices(ctrl_indices),
              qubit_indices(qubit_indices),
              bit_indices(bit_indices),
              loc(loc) {}
};

struct QasmCirc {
    size_t n_qubits;
    size_t n_bits;
    llvm::SmallVector<QasmInst> insts;

    QasmCirc() : n_qubits(0), n_bits(0) {}
};

std::optional<double> tryGetFloatVal(mlir::Value val) {
    mlir::FloatAttr float_attr;
    if (mlir::matchPattern(val, qcirc::m_CalcConstant(&float_attr))) {
        return float_attr.getValueAsDouble();
    } else {
        return {};
    }
}

// Convert a basic block (typically the body of a function) to a sequence of
// quantum gates
std::optional<QasmCirc> convertMLIRToGateSeq(mlir::Block &block) {
    QasmCirc circ;
    bool done = false;
    llvm::DenseMap<mlir::Value, unsigned int> edge_indices;
    std::queue<unsigned int> free_clean_qubit_indices;
    std::queue<unsigned int> free_dirty_qubit_indices;
    mlir::Type i1_type = mlir::IntegerType::get(block.front().getContext(), 1);

    for (mlir::Operation &op_ref : block) {
        mlir::Operation *op = &op_ref;

        qcirc::ArrayPackOp pack;
        if (done) {
            mlir::func::ReturnOp ret;
            if ((ret = llvm::dyn_cast<mlir::func::ReturnOp>(op))
                    && ret->getNumOperands() == 1
                    && (pack = ret->getOperand(0).getDefiningOp<qcirc::ArrayPackOp>())
                    && pack.getArray().getType().getElemType() == i1_type) {
                // Cool, just the final return statement
                continue;
            } else if (llvm::dyn_cast<qcirc::QfreeOp>(op)
                       || llvm::dyn_cast<qcirc::QfreeZeroOp>(op)) {
                // Allow qfrees after bitpack. Fallthrough
            } else {
                op->emitError("more instructions after bitpack");
                return {};
            }
        }

        if (qcirc::QallocOp qalloc = llvm::dyn_cast<qcirc::QallocOp>(op)) {
            if (free_clean_qubit_indices.empty()) {
                unsigned int new_idx;
                if (free_dirty_qubit_indices.empty()) {
                    new_idx = circ.n_qubits++;
                } else {
                    new_idx = free_dirty_qubit_indices.front();
                    free_dirty_qubit_indices.pop();
                    // Scrub this dirty qubit all clean
                    circ.insts.emplace_back(
                        "reset",
                        std::initializer_list<double>{},
                        std::initializer_list<unsigned int>{},
                        std::initializer_list<unsigned int>{new_idx},
                        std::initializer_list<unsigned int>{},
                        qalloc.getLoc());
                }
                free_clean_qubit_indices.push(new_idx);
            }
            unsigned int idx = free_clean_qubit_indices.front();
            free_clean_qubit_indices.pop();
            edge_indices[qalloc.getResult()] = idx;
        } else if (qcirc::QfreeOp qfree = llvm::dyn_cast<qcirc::QfreeOp>(op)) {
            if (!edge_indices.count(qfree.getQubit())) {
                op->emitError("bad edge into qfree");
                return {};
            }
            unsigned int idx = edge_indices[qfree.getQubit()];
            // Consume this qubit
            edge_indices.erase(qfree.getQubit());
            // Add to dirty freelist
            free_dirty_qubit_indices.push(idx);
        } else if (qcirc::QfreeZeroOp qfreez = llvm::dyn_cast<qcirc::QfreeZeroOp>(op)) {
            if (!edge_indices.count(qfreez.getQubit())) {
                op->emitError("bad edge into qfreez");
                return {};
            }
            unsigned int idx = edge_indices[qfreez.getQubit()];
            // Consume this qubit
            edge_indices.erase(qfreez.getQubit());
            // Add to clean freelist
            free_clean_qubit_indices.push(idx);
        } else if (qcirc::MeasureOp meas = llvm::dyn_cast<qcirc::MeasureOp>(op)) {
            if (!edge_indices.count(meas.getQubit())) {
                op->emitError("bad edge into measurement");
                return {};
            }
            unsigned int idx = edge_indices[meas.getQubit()];
            // Consume this qubit
            edge_indices.erase(meas.getQubit());
            edge_indices[meas.getQubitResult()] = idx;

            // Allocate a new bit for the measurement result
            unsigned int bit_idx = circ.n_bits++;
            circ.insts.emplace_back(
                "measure",
                std::initializer_list<double>{},
                std::initializer_list<unsigned int>{},
                std::initializer_list<unsigned int>{idx},
                std::initializer_list<unsigned int>{bit_idx},
                meas.getLoc());
            edge_indices[meas.getMeasResult()] = bit_idx;
        } else if (qcirc::Gate1QOp gate = llvm::dyn_cast<qcirc::Gate1QOp>(op)) {
            if (!edge_indices.count(gate.getQubit())) {
                op->emitError("bad edge into gate");
                return {};
            }
            unsigned int target_idx = edge_indices[gate.getQubit()];
            // Consume this qubit
            edge_indices.erase(gate.getQubit());

            std::vector<unsigned int> ctrls;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                mlir::Value ctrl = gate.getControls()[i];
                if (!edge_indices.count(ctrl)) {
                    op->emitError("bad edge into ctrl of gate");
                    return {};
                }
                unsigned int ctrl_idx = edge_indices[ctrl];
                // Consume this qubit
                edge_indices.erase(ctrl);
                ctrls.push_back(ctrl_idx);
            }
            circ.insts.emplace_back(
                qcirc::stringifyGate1Q(gate.getGate()).lower(),
                std::initializer_list<double>{},
                ctrls,
                std::initializer_list<unsigned int>{target_idx},
                std::initializer_list<unsigned int>{},
                gate.getLoc());

            edge_indices[gate.getResult()] = target_idx;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                edge_indices[gate.getControlResults()[i]] = ctrls[i];
            }
        } else if (qcirc::Gate1Q1POp gate = llvm::dyn_cast<qcirc::Gate1Q1POp>(op)) {
            if (!edge_indices.count(gate.getQubit())) {
                op->emitError("bad edge into gate");
                return {};
            }
            unsigned int target_idx = edge_indices[gate.getQubit()];
            // Consume this qubit
            edge_indices.erase(gate.getQubit());

            std::vector<unsigned int> ctrls;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                mlir::Value ctrl = gate.getControls()[i];
                if (!edge_indices.count(ctrl)) {
                    op->emitError("bad edge into ctrl of gate");
                    return {};
                }
                unsigned int ctrl_idx = edge_indices[ctrl];
                // Consume this qubit
                edge_indices.erase(ctrl);
                ctrls.push_back(ctrl_idx);
            }

            auto param = tryGetFloatVal(gate.getParam());
            if (!param) {
                op->emitError("nonconstant parameter for gate1q1p");
                return {};
            }
            double parameter = param.value();

            circ.insts.emplace_back(
                qcirc::stringifyGate1Q1P(gate.getGate()).lower(),
                std::initializer_list<double>{parameter},
                ctrls,
                std::initializer_list<unsigned int>{target_idx},
                std::initializer_list<unsigned int>{},
                gate.getLoc());

            edge_indices[gate.getResult()] = target_idx;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                edge_indices[gate.getControlResults()[i]] = ctrls[i];
            }
        } else if (qcirc::Gate1Q3POp gate = llvm::dyn_cast<qcirc::Gate1Q3POp>(op)) {
            if (!edge_indices.count(gate.getQubit())) {
                op->emitError("bad edge into gate");
                return {};
            }
            unsigned int target_idx = edge_indices[gate.getQubit()];
            // Consume this qubit
            edge_indices.erase(gate.getQubit());

            std::vector<unsigned int> ctrls;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                mlir::Value ctrl = gate.getControls()[i];
                if (!edge_indices.count(ctrl)) {
                    op->emitError("bad edge into ctrl of gate");
                    return {};
                }
                unsigned int ctrl_idx = edge_indices[ctrl];
                // Consume this qubit
                edge_indices.erase(ctrl);
                ctrls.push_back(ctrl_idx);
            }

            // TODO: use a function here instead of copypasta
            auto param1 = tryGetFloatVal(gate.getFirstParam()),
                 param2 = tryGetFloatVal(gate.getSecondParam()),
                 param3 = tryGetFloatVal(gate.getThirdParam());
            if (!param1 || !param2 || !param3) {
                op->emitError("nonconstant parameter for gate1q3p");
                return {};
            }
            double parameter1 = param1.value(),
                   parameter2 = param2.value(),
                   parameter3 = param3.value();

            circ.insts.emplace_back(
                qcirc::stringifyGate1Q3P(gate.getGate()).lower(),
                std::initializer_list<double>{parameter1, parameter2, parameter3},
                ctrls,
                std::initializer_list<unsigned int>{target_idx},
                std::initializer_list<unsigned int>{},
                gate.getLoc());

            edge_indices[gate.getResult()] = target_idx;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                edge_indices[gate.getControlResults()[i]] = ctrls[i];
            }
        } else if (qcirc::Gate2QOp gate = llvm::dyn_cast<qcirc::Gate2QOp>(op)) {
            if (!edge_indices.count(gate.getLeftQubit())) {
                op->emitError("bad edge into left qubit of gate");
                return {};
            }
            unsigned int left_idx = edge_indices[gate.getLeftQubit()];
            // Consume this qubit
            edge_indices.erase(gate.getLeftQubit());
            if (!edge_indices.count(gate.getRightQubit())) {
                op->emitError("bad edge into right qubit of gate");
                return {};
            }
            unsigned int right_idx = edge_indices[gate.getRightQubit()];
            // Consume this qubit
            edge_indices.erase(gate.getRightQubit());

            std::vector<unsigned int> ctrls;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                mlir::Value ctrl = gate.getControls()[i];
                if (!edge_indices.count(ctrl)) {
                    op->emitError("bad edge into ctrl of gate");
                    return {};
                }
                unsigned int ctrl_idx = edge_indices[ctrl];
                ctrls.push_back(ctrl_idx);
            }
            circ.insts.emplace_back(
                qcirc::stringifyGate2Q(gate.getGate()).lower(),
                std::initializer_list<double>{},
                ctrls,
                std::initializer_list<unsigned int>{left_idx, right_idx},
                std::initializer_list<unsigned int>{},
                gate.getLoc());

            edge_indices[gate.getLeftResult()] = left_idx;
            edge_indices[gate.getRightResult()] = right_idx;
            for (unsigned int i = 0; i < gate.getControls().size(); i++) {
                edge_indices[gate.getControlResults()[i]] = ctrls[i];
            }
        } else if ((pack = llvm::dyn_cast<qcirc::ArrayPackOp>(op))
                   && pack.getArray().getType().getElemType() == i1_type) {
            std::unordered_map<unsigned int, unsigned int> permutation;
            for (unsigned int i = 0; i < pack.getElems().size(); i++) {
                mlir::Value bit = pack.getElems()[i];
                if (!edge_indices.count(bit)) {
                    op->emitError("bad edge into bit of bitpack");
                    return {};
                }
                permutation[edge_indices[bit]] = i;
                // Consume this bit
                edge_indices.erase(bit);
            }

            for (unsigned int i = 0; i < circ.insts.size(); i++) {
                // Rewrite classical bits to get the ordering right
                for (unsigned int j = 0; j < circ.insts[i].bit_indices.size(); j++) {
                    circ.insts[i].bit_indices[j] = permutation[circ.insts[i].bit_indices[j]];
                }
            }
            done = true;
        } else if (llvm::isa<qcirc::CalcOp>(op)) {
            // Ignore this guy. A constant will be extracted from it as needed
            // when processing gate1q1ps and gate1p3ps above.
        } else {
            op->emitError("encountered unknown op in qasm generation");
            return {};
        }
    }

    if (!edge_indices.empty()) {
        for (auto &kv : edge_indices) {
            mlir::Value val = kv.first;
            if (mlir::Operation *op = val.getDefiningOp()) {
                op->emitError("edges still live after walk. need to run canonicalizer?");
            }
        }
        return {};
    }

    return circ;
}

void printGatesAsOpenQasm3(std::ostream &os, QasmCirc &circ, bool print_locs) {
    os << "OPENQASM 3.0;\n";
    os << "include \"stdgates.inc\";\n";
    os << "qreg q[" << circ.n_qubits << "];\n";
    os << "creg c[" << circ.n_bits << "];\n";
    for (unsigned int i = 0; i < circ.insts.size(); i++) {
        QasmInst &inst = circ.insts[i];

        if (!inst.ctrl_indices.empty()) {
            os << "ctrl(" << inst.ctrl_indices.size() << ") @ ";
        }
        // Handle some special cases for gates not in stdgates.inc, at
        // least according to the OpenQASM 3 TQC paper
        if (inst.name == "u") {
            os << "U";
        } else if (inst.name == "sxdg") {
            os << "inv @ sx";
        } else {
            os << inst.name;
        }

        if (!inst.params.empty()) {
            os << "(";
            for (unsigned int j = 0; j < inst.params.size(); j++) {
                if (j) {
                    os << ",";
                }
                os << inst.params[j];
            }
            os << ")";
        }
        os << " ";

        for (unsigned int j = 0; j < inst.ctrl_indices.size(); j++) {
            if (j) {
                os << ", ";
            }
            os << "q[" << inst.ctrl_indices[j] << "]";
        }

        for (unsigned int j = 0; j < inst.qubit_indices.size(); j++) {
            if (j || !inst.ctrl_indices.empty()) {
                os << ", ";
            }
            os << "q[" << inst.qubit_indices[j] << "]";
        }

        for (unsigned int j = 0; j < inst.bit_indices.size(); j++) {
            if (j) {
                os << ", ";
            } else {
                os << " -> ";
            }
            os << "c[" << inst.bit_indices[j] << "]";
        }

        os << ";";

        if (print_locs) {
            os << " // ";
            std::string ret;
            llvm::raw_string_ostream ostream(ret);
            inst.loc.print(ostream);
            os << ret;
        }

        os << "\n";
    }
}

} // namespace

namespace qcirc {

// Overall idea here is to convert straight-line value-semantics code in a
// mlir::func::FuncOp into a sequence of gates with indices. In the process,
// qubits are statically allocated. The overall idea is to keep a map of
// mlir::Values to qubit indices.
mlir::LogicalResult generateQasm(mlir::func::FuncOp func_op, bool print_locs, std::string &result) {
    if (func_op->getNumRegions() != 1
            || !func_op.getBody().hasOneBlock()) {
        func_op->emitError("expected a FuncOp with 1 block");
        return mlir::failure();
    }
    mlir::Block &body_block = func_op.getBody().front();
    std::optional<QasmCirc> circ = convertMLIRToGateSeq(body_block);
    if (!circ) {
        return mlir::failure();
    }

    std::ostringstream ss;
    printGatesAsOpenQasm3(ss, *circ, print_locs);

    result = ss.str();
    return mlir::success();
}

} // namespace qcirc
