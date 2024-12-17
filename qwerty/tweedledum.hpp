#ifndef TWEEDLEDUM_H
#define TWEEDLEDUM_H

#include "tweedledum/IR/Circuit.h"
#include "mockturtle/networks/xag.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

// This is a wrapper around Tweedledum.
// This code is unique in that it is called by both the AST-based frontend and
// the MLIR end of things too. We do not want the MLIR guts to depend on the
// AST frontend stuff, so we cannot #include "ast.hpp" here, nor can we throw
// exceptions. (In fact, this is compiled as a separate standalone library,
// libqwutil.)

struct TweedledumCircuit {
    tweedledum::Circuit circ;
    size_t n_total_qubits, n_data_qubits, n_ancilla_qubits;

    TweedledumCircuit(tweedledum::Circuit raw_circ)
                     : circ(cleanupCircuit(raw_circ)),
                       n_total_qubits(circ.num_qubits()),
                       n_data_qubits(numDataQubits(circ)),
                       n_ancilla_qubits(n_total_qubits - n_data_qubits) {}

    // Create a TweedledumCircuit from a mockturtle netlist
    static TweedledumCircuit fromNetlist(mockturtle::xag_network &raw_net);
    // Create a TweedledumCircuit from a classical permutation
    static TweedledumCircuit fromPermutation(std::vector<uint32_t> &perm);

    static size_t numDataQubits(tweedledum::Circuit &circ);
    static tweedledum::Circuit cleanupCircuit(tweedledum::Circuit &circ);

    // Transpile to QCirc IR, putting the result in a FuncOp with the specified
    // symbol name (`name')
    qwerty::FuncOp toFuncOp(mlir::OpBuilder &builder,
                            mlir::Location loc,
                            mlir::ModuleOp module,
                            const std::string name);

    // Similar to toFuncOp(), except the QCirc ops are created inline wherever
    // the builder's insertion point is, without creating a new FuncOp
    void toQCircInline(mlir::OpBuilder &builder,
                       mlir::Location loc,
                       llvm::SmallVectorImpl<mlir::Value> &qubits,
                       size_t qubit_idx) {
        llvm::SmallVector<mlir::Value> no_controls;
        toQCircInline(builder, loc, no_controls, qubits, qubit_idx);
    }

    // Same as toQCircInline() except also adds controls to each gate
    void toQCircInline(mlir::OpBuilder &builder,
                       mlir::Location loc,
                       llvm::SmallVectorImpl<mlir::Value> &control_qubits,
                       llvm::SmallVectorImpl<mlir::Value> &qubits,
                       size_t qubit_idx);

    // Dump this circuit in ASCII diagrammatic form to a file. Useful for
    // debugging
    void toFile(const std::string base_name);
};

#endif // TWEEDLEDUM_H
