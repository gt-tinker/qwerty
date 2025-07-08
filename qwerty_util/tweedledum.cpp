#include "util.hpp"

#include <fstream>

#include "mockturtle/algorithms/cleanup.hpp"
#include "tweedledum/Passes/Decomposition/parity_decomp.h"
#include "tweedledum/Synthesis/transform_synth.h"
#include "tweedledum/Synthesis/xag_synth.h"
#include "tweedledum/Utils/Classical/xag_optimize.h"
#include "tweedledum/Utils/Visualization/string_utf8.h"
#include "tweedledum/Operators/Standard.h"
#include "tweedledum/Operators/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"

#include "QCirc/IR/QCircOps.h"
#include "QCirc/IR/QCircAttributes.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/IR/QwertyTypes.h"

#include "tweedledum.hpp"

namespace {
mlir::Value wrapFloatConst(mlir::OpBuilder &builder,
                           mlir::Location loc,
                           double theta) {
    qcirc::CalcOp calc = builder.create<qcirc::CalcOp>(loc, builder.getF64Type(), mlir::ValueRange());
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        // Sets insertion point to end of this block
        mlir::Block *calc_block = builder.createBlock(
            &calc.getRegion(), {}, mlir::TypeRange(), {});
        assert(!calc_block->getNumArguments());
        mlir::Value ret = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getF64FloatAttr(theta)).getResult();
        builder.create<qcirc::CalcYieldOp>(loc, ret);
    }
    mlir::ValueRange calc_results = calc.getResults();
    assert(calc_results.size() == 1);
    return calc_results[0];
}

} // namespace

TweedledumCircuit TweedledumCircuit::fromNetlist(
        mockturtle::xag_network &raw_net) {
    mockturtle::xag_network net = mockturtle::cleanup_dangling(raw_net);
    tweedledum::xag_optimize(net);
    return TweedledumCircuit(tweedledum::xag_synth(net));
}

TweedledumCircuit TweedledumCircuit::fromPermutation(
        std::vector<uint32_t> &perm) {
    return TweedledumCircuit(tweedledum::transform_synth(perm));
}

// Best way I can find to do this. God bless
size_t TweedledumCircuit::numDataQubits(tweedledum::Circuit &circ) {
    for (size_t i = 0; i < circ.num_qubits(); i++) {
        std::string_view name = circ.name(circ.qubit(i));
        if (name.substr(0, 3) == "__a") {
            return i;
        }
    }
    return circ.num_qubits();
}

tweedledum::Circuit TweedledumCircuit::cleanupCircuit(
        tweedledum::Circuit &circ) {
    return tweedledum::parity_decomp(circ);
}

void TweedledumCircuit::toFile(const std::string base_name) {
    std::ofstream stream("tweedledum_circ_" + base_name + ".txt");
    // Use 100 columns because we are big boys with big boy monitors
    stream << tweedledum::to_string_utf8(circ, /*max_rows=*/100);
    stream.close();
}

void TweedledumCircuit::toQCircInline(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &control_qubits,
        llvm::SmallVectorImpl<mlir::Value> &raw_qubits,
        size_t qubit_idx) {
    assert(raw_qubits.size() - qubit_idx >= n_data_qubits
           && "Too few qubits for the mission here buddy");
    llvm::SmallVector<mlir::Value> qubits(
        raw_qubits.begin() + qubit_idx,
        raw_qubits.begin() + qubit_idx + n_data_qubits);

    for (size_t i = 0; i < n_ancilla_qubits; i++) {
        qubits.push_back(builder.create<qcirc::QallocOp>(loc).getResult());
    }

    circ.foreach_instruction([&](const tweedledum::Instruction &inst) {
        llvm::SmallVector<mlir::Value> controls(
            control_qubits.begin(), control_qubits.end());
        for (size_t i = 0; i < inst.num_controls(); i++) {
            tweedledum::Qubit qubit = inst.control(i);
            size_t qubit_idx = qubit.uid();
            if (qubit.polarity() == tweedledum::Qubit::Polarity::negative) {
                // Controlled-on-zero
                qubits[qubit_idx] = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                    qubits[qubit_idx]).getResult();
            }
            controls.push_back(qubits[qubit_idx]);
        }
        mlir::ValueRange controlResults;

        if (inst.num_targets() == 2) {
            // Swap gates
            if (inst.name() == "swap") {
                uint64_t leftIdx = inst.target(0).uid();
                uint64_t rightIdx = inst.target(1).uid();
                auto output = builder.create<qcirc::Gate2QOp>(
                    loc, qcirc::Gate2Q::Swap, controls,
                    qubits[leftIdx], qubits[rightIdx]);
                qubits[leftIdx] = output.getLeftResult();
                qubits[rightIdx] = output.getRightResult();
                controlResults = output.getControlResults();
            } else {
                assert(0 && "Unknown two-qubit gate");
            }
        } else if (inst.num_targets() == 1) {
            // Controlled gates
            qcirc::Gate1Q gate = llvm::StringSwitch<qcirc::Gate1Q>(inst.name())
                                   .Case("x",    qcirc::Gate1Q::X)
                                   .Case("y",    qcirc::Gate1Q::Y)
                                   .Case("z",    qcirc::Gate1Q::Z)
                                   .Case("h",    qcirc::Gate1Q::H)
                                   .Case("s",    qcirc::Gate1Q::S)
                                   .Case("sdg",  qcirc::Gate1Q::Sdg)
                                   .Case("sx",   qcirc::Gate1Q::Sx)
                                   .Case("sxdg", qcirc::Gate1Q::Sxdg)
                                   .Case("t",    qcirc::Gate1Q::T)
                                   .Case("tdg",  qcirc::Gate1Q::Tdg)
                                   .Default((qcirc::Gate1Q)-1);

            qcirc::Gate1Q1P gate1p = llvm::StringSwitch<qcirc::Gate1Q1P>(inst.name())
                                      .Case("p",  qcirc::Gate1Q1P::P)
                                      .Case("rx", qcirc::Gate1Q1P::Rx)
                                      .Case("ry", qcirc::Gate1Q1P::Ry)
                                      .Case("rz", qcirc::Gate1Q1P::Rz)
                                      .Default((qcirc::Gate1Q1P)-1);

            qcirc::Gate1Q3P gate3p = llvm::StringSwitch<qcirc::Gate1Q3P>(inst.name())
                                       .Case("u", qcirc::Gate1Q3P::U)
                                       .Default((qcirc::Gate1Q3P)-1);

            uint64_t idx = inst.target(0).uid();
            mlir::Value inputQubit = qubits[idx];
            mlir::Value outputQubit;

            if (gate != (qcirc::Gate1Q)-1) {
                auto output = builder.create<qcirc::Gate1QOp>(
                    loc, gate, controls, inputQubit);
                outputQubit = output.getResult();
                controlResults = output.getControlResults();
            } else if (gate1p != (qcirc::Gate1Q1P)-1) {
                mlir::Value param = wrapFloatConst(builder, loc, tweedledum::rotation_angle(inst).value());
                auto output = builder.create<qcirc::Gate1Q1POp>(
                    loc, gate1p, param, controls, inputQubit);
                outputQubit = output.getResult();
                controlResults = output.getControlResults();
            } else if (gate3p != (qcirc::Gate1Q3P)-1) {
                const tweedledum::Op::U &u = inst.cast<tweedledum::Op::U>();
                mlir::Value firstParam = wrapFloatConst(builder, loc, u.theta());
                mlir::Value secondParam = wrapFloatConst(builder, loc, u.phi());
                mlir::Value thirdParam = wrapFloatConst(builder, loc, u.lambda());

                auto output = builder.create<qcirc::Gate1Q3POp>(
                    loc, gate3p, firstParam, secondParam, thirdParam,
                    controls, inputQubit);
                outputQubit = output.getResult();
                controlResults = output.getControlResults();
            } else {
                assert(0 && "Unknown gate");
            }

            qubits[idx] = outputQubit;
        } else {
            assert(0 && "Unsupported number of target qubits");
        }

        size_t n_fixed_controls = control_qubits.size();
        control_qubits.clear();
        control_qubits.append(controlResults.begin(),
                              controlResults.begin() + n_fixed_controls);

        for (size_t i = 0; i < inst.num_controls(); i++) {
            tweedledum::Qubit qubit = inst.control(i);
            size_t qubit_idx = qubit.uid();
            if (qubit.polarity() == tweedledum::Qubit::Polarity::negative) {
                // Controlled-on-zero, so undo the X we added earlier
                qubits[qubit_idx] = builder.create<qcirc::Gate1QOp>(
                    loc, qcirc::Gate1Q::X, mlir::ValueRange(),
                    controlResults[n_fixed_controls + i]).getResult();
            } else {
                qubits[qubit_idx] = controlResults[n_fixed_controls + i];
            }
        }
    });

    double global_phase = circ.global_phase();
    if (std::abs(global_phase) >= ATOL) {
        // Undo global phase just in case we end up using a controlled
        // version of this
        double phase = -global_phase;
        mlir::Value phase_val = wrapFloatConst(builder, loc, phase);
        if (control_qubits.empty()) {
            // Adjust 1 amplitude
            qubits[0] = builder.create<qcirc::Gate1Q1POp>(loc, qcirc::Gate1Q1P::P, phase_val, mlir::ValueRange(), qubits[0]).getResult();
            // Adjust 0 amplitude
            qubits[0] = builder.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[0]).getResult();
            qubits[0] = builder.create<qcirc::Gate1Q1POp>(loc, qcirc::Gate1Q1P::P, phase_val, mlir::ValueRange(), qubits[0]).getResult();
            qubits[0] = builder.create<qcirc::Gate1QOp>(loc, qcirc::Gate1Q::X, mlir::ValueRange(), qubits[0]).getResult();
        } else {
            llvm::SmallVector<mlir::Value> gp_controls(
                control_qubits.begin()+1,
                control_qubits.end());
            qcirc::Gate1Q1POp gp = builder.create<qcirc::Gate1Q1POp>(
                loc, qcirc::Gate1Q1P::P, phase_val,
                gp_controls, control_qubits[0]);
            control_qubits.clear();
            control_qubits.push_back(gp.getResult());
            control_qubits.append(gp.getControlResults().begin(),
                                  gp.getControlResults().end());
        }
    }

    for (size_t i = 0; i < n_ancilla_qubits; i++) {
        builder.create<qcirc::QfreeZeroOp>(loc, qubits[n_data_qubits + i]);
    }
    qubits.pop_back_n(n_ancilla_qubits);

    for (size_t i = 0; i < n_data_qubits; i++) {
        raw_qubits[qubit_idx + i] = qubits[i];
    }
}

qwerty::FuncOp TweedledumCircuit::toFuncOp(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::ModuleOp module,
        const std::string name) {
    auto qbundle_type = builder.getType<qwerty::QBundleType>(n_data_qubits);
    auto func_type = builder.getType<qwerty::FunctionType>(
        builder.getFunctionType({qbundle_type}, {qbundle_type}),
        /*reversible=*/true);

    builder.setInsertionPointToEnd(module.getBody());
    auto func = builder.create<qwerty::FuncOp>(loc, name, func_type);
    // Sets insert point to end of this block
    mlir::Block *block = builder.createBlock(&func.getBody(), {}, {qbundle_type}, {loc});
    auto unpacked = builder.create<qwerty::QBundleUnpackOp>(loc, block->getArgument(0)).getQubits();

    llvm::SmallVector<mlir::Value> qubits(unpacked);
    toQCircInline(builder, loc, qubits, 0);

    auto repacked = builder.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();
    builder.create<qwerty::ReturnOp>(loc, repacked);

    // Set the function symbol as private so that inlining will prune it. The
    // programmer shouldn't be calling a @classical kernel directly; it should
    // only be called inside a @qpu kernel
    func.setPrivate();

    if (mlir::failed(mlir::verify(func))) {
        return nullptr;
    } else {
        return func;
    }
}
