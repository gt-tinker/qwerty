#include <queue>

#include "mlir/IR/SymbolTable.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "QCirc/IR/QCircOps.h"
#include "CCirc/IR/CCircOps.h"
#include "tweedledum.hpp"
#include "PassDetail.h"

// This is a pass that converts all qwerty.embed_* ops that reference classical
// circuits into qwerty.func_const ops referencing qwerty.funcs that contain
// quantum circuits.

namespace {

// Create an XOR embedding of a classical circuit.
qwerty::FuncOp synthXor(
        mlir::RewriterBase &rewriter, ccirc::CircuitOp circ, mlir::Location loc) {
    qwerty::FunctionType func_ty = qwerty::EmbedXorOp::getQwertyFuncTypeOf(circ);
    std::string embed_func_name = circ.getSymName().str() + "__xor";

    rewriter.setInsertionPointAfter(circ);
    qwerty::FuncOp new_func = rewriter.create<qwerty::FuncOp>(loc, embed_func_name, func_ty);
    new_func.setPrivate();
    mlir::Block *block = rewriter.createBlock(
        &new_func.getBody(), {}, func_ty.getFunctionType().getInputs(), {loc});

    assert(block->getNumArguments() == 1
           && "Wrong number of arguments for embed block");
    qwerty::QBundleUnpackOp unpack = rewriter.create<qwerty::QBundleUnpackOp>(
        loc, block->getArgument(0));
    llvm::SmallVector<mlir::Value> qubits(unpack.getQubits());

    TweedledumCircuit::fromCCirc(circ).toQCircInline(rewriter, loc, qubits, 0);

    mlir::Value packed =
        rewriter.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();
    rewriter.create<qwerty::ReturnOp>(loc, packed);
    return new_func;
}

// Create a sign (phase kickback) embedding of a classical circuit.
qwerty::FuncOp synthSign(
        mlir::RewriterBase &rewriter,
        ccirc::CircuitOp circ,
        qwerty::FuncOp xor_func,
        mlir::Location loc) {
    size_t in_dim = circ.inDim();
    size_t out_dim = circ.outDim();

    // if the circuit was bit[m] -> bit[n], then the embedding func type should
    // be rev_qfunc[m]
    mlir::Type qbundle_type = rewriter.getType<qwerty::QBundleType>(in_dim);
    qwerty::FunctionType embed_func_ty =
        rewriter.getType<qwerty::FunctionType>(
            rewriter.getFunctionType({qbundle_type}, {qbundle_type}),
            /*reversible=*/true);
    std::string embed_func_name = circ.getSymName().str() + "__sign";

    rewriter.setInsertionPointAfter(xor_func);
    qwerty::FuncOp new_func = rewriter.create<qwerty::FuncOp>(loc, embed_func_name, embed_func_ty);
    // Sets insert point to end of this block
    mlir::Block *block = rewriter.createBlock(&new_func.getBody(), {}, {qbundle_type}, {loc});
    assert(block->getNumArguments() == 1 && "Expected 1 argument to sign block");

    qwerty::BasisAttr basis = rewriter.getAttr<qwerty::BasisAttr>(
        std::initializer_list<qwerty::BasisElemAttr>{
            rewriter.getAttr<qwerty::BasisElemAttr>(
                rewriter.getAttr<qwerty::BasisVectorListAttr>(
                    std::initializer_list<qwerty::BasisVectorAttr>{
                        rewriter.getAttr<qwerty::BasisVectorAttr>(
                            qwerty::PrimitiveBasis::X, qwerty::Eigenstate::MINUS,
                            out_dim, /*hasPhase=*/false)}))});
    mlir::Value zero = rewriter.create<qwerty::QBundlePrepOp>(
        loc, qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::PLUS, out_dim).getResult();
    mlir::Value minus = rewriter.create<qwerty::QBundleInitOp>(
        loc, basis, mlir::ValueRange(), zero).getQbundleOut();
    mlir::ValueRange minus_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, minus).getQubits();

    mlir::Value qbundle_arg = block->getArgument(0);
    mlir::ValueRange arg_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, qbundle_arg).getQubits();
    llvm::SmallVector<mlir::Value> all_merged(arg_unpacked.begin(), arg_unpacked.end());
    all_merged.append(minus_unpacked.begin(), minus_unpacked.end());
    mlir::Value repacked = rewriter.create<qwerty::QBundlePackOp>(loc, all_merged).getQbundle();

    mlir::ValueRange results = rewriter.create<qwerty::CallOp>(loc, xor_func, repacked).getResults();
    mlir::ValueRange results_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, results).getQubits();

    llvm::SmallVector<mlir::Value> output_qubits(results_unpacked.begin(), results_unpacked.begin()+in_dim);
    mlir::Value output = rewriter.create<qwerty::QBundlePackOp>(loc, output_qubits).getQbundle();
    llvm::SmallVector<mlir::Value> uncompute_qubits(results_unpacked.begin()+in_dim, results_unpacked.end());
    mlir::Value to_uncompute = rewriter.create<qwerty::QBundlePackOp>(loc, uncompute_qubits).getQbundle();

    mlir::Value to_discard = rewriter.create<qwerty::QBundleDeinitOp>(
        loc, basis, mlir::ValueRange(), to_uncompute).getQbundleOut();
    rewriter.create<qwerty::QBundleDiscardZeroOp>(loc, to_discard);
    rewriter.create<qwerty::ReturnOp>(loc, output);

    // Has non-classical inputs
    new_func.setPrivate();
    return new_func;
}

qwerty::FuncOp synthInPlace(
        mlir::RewriterBase &rewriter,
        ccirc::CircuitOp fwd_circ,
        qwerty::FuncOp fwd_xor_func,
        mlir::Location loc) {
    // Step one: we need an inverse function
    rewriter.setInsertionPointAfter(fwd_xor_func);
    std::string inv_circ_name = fwd_circ.getSymName().str() + "__inv";
    ccirc::CircuitOp inv_circ = fwd_circ.buildInverseCircuit(rewriter, loc, inv_circ_name);
    qwerty::FuncOp inv_xor_func = synthXor(rewriter, inv_circ, loc);

    // Step two: we need to create a new function that uses Bennett's trick

    size_t dim = fwd_circ.inDim();
    assert(dim == fwd_circ.outDim()
           && "Reversible function's input dimension must match its output "
              "dimension");

    // If the classical func was bit[n] -> bit[n], then the in-place embedding
    // should have function type rev_qfunc[n].
    mlir::Type qbundle_type = rewriter.getType<qwerty::QBundleType>(dim);
    qwerty::FunctionType new_func_ty =
        rewriter.getType<qwerty::FunctionType>(
            rewriter.getFunctionType({qbundle_type}, {qbundle_type}),
            /*reversible=*/true);

    std::string embed_func_name = fwd_circ.getSymName().str() + "__inplace";

    // This will just be a stub function that calls the XOR embedding
    rewriter.setInsertionPointAfter(inv_xor_func);
    qwerty::FuncOp new_func = rewriter.create<qwerty::FuncOp>(loc, embed_func_name, new_func_ty);
    // Sets insert point to end of this block
    mlir::Block *block = rewriter.createBlock(&new_func.getBody(), {}, {qbundle_type}, {loc});
    assert(block->getNumArguments() == 1 && "Wrong number of arguments to inplace block");

    mlir::Value zeros = rewriter.create<qwerty::QBundlePrepOp>(
            loc, qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::PLUS, dim).getResult();
    mlir::ValueRange zeros_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, zeros).getQubits();
    mlir::Value qbundle_arg = block->getArgument(0);
    mlir::ValueRange arg_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, qbundle_arg).getQubits();
    llvm::SmallVector<mlir::Value> all_merged(arg_unpacked.begin(), arg_unpacked.end());
    all_merged.append(zeros_unpacked.begin(), zeros_unpacked.end());
    mlir::Value repacked = rewriter.create<qwerty::QBundlePackOp>(loc, all_merged).getQbundle();

    mlir::ValueRange results = rewriter.create<qwerty::CallOp>(loc, fwd_xor_func, repacked).getResults();
    mlir::ValueRange results_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, results).getQubits();

    //llvm::SmallVector<mlir::Value> qubits(results_unpacked);
    //for (size_t i = 0; i < dim; i++) {
    //    // Swap by renaming
    //    std::swap(qubits[i], qubits[i + dim]);
    //}
    //mlir::Value swapped_repacked = rewriter.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();

    llvm::SmallVector<mlir::Value> qubits(results_unpacked);
    for (size_t i = 0; i < dim; i++) {
        qcirc::Gate2QOp swap = rewriter.create<qcirc::Gate2QOp>(
            loc, qcirc::Gate2Q::Swap, mlir::ValueRange(), qubits[i], qubits[i + dim]);
        qubits[i] = swap.getLeftResult();
        qubits[i + dim] = swap.getRightResult();
    }
    mlir::Value swapped_repacked = rewriter.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();

    mlir::ValueRange inv = rewriter.create<qwerty::CallOp>(loc, inv_xor_func, swapped_repacked).getResults();
    assert(inv.size() == 1 && "XOR embedding should return 1 value");
    mlir::ValueRange inv_unpacked = rewriter.create<qwerty::QBundleUnpackOp>(loc, inv[0]).getQubits();

    llvm::SmallVector<mlir::Value> discard_qubits(inv_unpacked.begin()+dim, inv_unpacked.end());
    mlir::Value to_discard = rewriter.create<qwerty::QBundlePackOp>(loc, discard_qubits).getQbundle();
    rewriter.create<qwerty::QBundleDiscardZeroOp>(loc, to_discard);

    llvm::SmallVector<mlir::Value> output_qubits(inv_unpacked.begin(), inv_unpacked.begin()+dim);
    mlir::Value output = rewriter.create<qwerty::QBundlePackOp>(loc, output_qubits).getQbundle();
    rewriter.create<qwerty::ReturnOp>(loc, output);

    // Has non-classical inputs
    new_func.setPrivate();
    return new_func;
}

struct SynthEmbedsPass
        : public qwerty::SynthEmbedsBase<SynthEmbedsPass> {
    void runOnOperation() override {
        mlir::ModuleOp mod = getOperation();

        llvm::SmallVector<ccirc::CircuitOp> worklist;
        for (ccirc::CircuitOp circ :
                mod.getBodyRegion().getOps<ccirc::CircuitOp>()) {
            worklist.push_back(circ);
        }

        llvm::DenseMap<ccirc::CircuitOp,
                       llvm::SmallVector<mlir::Operation *>> modify_queues;

        while (!worklist.empty()) {
            ccirc::CircuitOp circ = worklist.back();
            worklist.pop_back();

            std::optional<mlir::SymbolTable::UseRange> uses_opt =
                mlir::SymbolTable::getSymbolUses(circ.getSymNameAttr(), mod);
            if (!uses_opt.has_value()) {
                continue;
            }

            for (const mlir::SymbolTable::SymbolUse &use : uses_opt.value()) {
                modify_queues[circ].push_back(use.getUser());
            }
        }

        // Now we modify the IR.
        mlir::IRRewriter rewriter(&getContext());
        for (auto [circ, users] : modify_queues) {
            mlir::Location loc = circ.getLoc();
            qwerty::FuncOp xor_func, sign_func, inplace_func;

            for (mlir::Operation *user : users) {
                if (qwerty::EmbedXorOp embed_xor = llvm::dyn_cast<qwerty::EmbedXorOp>(user)) {
                    if (!xor_func) {
                        xor_func = synthXor(rewriter, circ, loc);
                    }
                    rewriter.setInsertionPoint(embed_xor);
                    rewriter.replaceOpWithNewOp<qwerty::FuncConstOp>(embed_xor, xor_func);
                } else if (qwerty::EmbedSignOp embed_sign = llvm::dyn_cast<qwerty::EmbedSignOp>(user)) {
                    if (!sign_func) {
                        // We still need the XOR embedding
                        if (!xor_func) {
                            xor_func = synthXor(rewriter, circ, loc);
                        }
                        sign_func = synthSign(rewriter, circ, xor_func, loc);
                    }
                    rewriter.setInsertionPoint(embed_sign);
                    rewriter.replaceOpWithNewOp<qwerty::FuncConstOp>(embed_sign, sign_func);
                } else if (qwerty::EmbedInPlaceOp embed_inplace = llvm::dyn_cast<qwerty::EmbedInPlaceOp>(user)) {
                    if (!inplace_func) {
                        // We use the XOR embedding as the forward func
                        if (!xor_func) {
                            xor_func = synthXor(rewriter, circ, loc);
                        }
                        inplace_func = synthInPlace(rewriter, circ, xor_func, loc);
                    }
                    rewriter.setInsertionPoint(embed_inplace);
                    rewriter.replaceOpWithNewOp<qwerty::FuncConstOp>(embed_inplace, inplace_func);
                } else {
                    assert(!llvm::isa_and_present<qwerty::QwertyDialect>(user->getDialect())
                           && "Missing handling of embed op");

                    // Otherwise, could be a circuit calling another circuit.
                    // That's fine.
                }
            }
        }

        // Pass pipeline should run symbol-dce afterward to get rid of ccirc.circuits
    }
};

} // namespace

std::unique_ptr<mlir::Pass> qwerty::createSynthEmbedsPass() {
    return std::make_unique<SynthEmbedsPass>();
}
