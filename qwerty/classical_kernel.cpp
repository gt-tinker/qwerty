#include "mockturtle/io/write_dot.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "QCirc/IR/QCircOps.h"
#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/IR/QwertyTypes.h"

#include "defs.hpp"
#include "ast.hpp"
#include "mlir_handle.hpp"
#include "tweedledum.hpp"
#include "ast_visitor.hpp"

namespace {
qwerty::FuncOp constructPhaseEmbedding(
        MlirHandle &handle,
        mlir::Location loc,
        qwerty::FuncOp xor_embed,
        std::string stub_name,
        FuncType &classical_type) {
    // ===> Save insertion point
    auto old_insertpt = handle.builder.saveInsertionPoint();

    std::unique_ptr<Type> canon_classical_type = classical_type.canonicalize();
    FuncType &canon_func_type = dynamic_cast<FuncType &>(*canon_classical_type);
    BitType &lhs = dynamic_cast<BitType &>(*canon_func_type.lhs);
    BitType &rhs = dynamic_cast<BitType &>(*canon_func_type.rhs);
    assert(lhs.dim->isConstant() && rhs.dim->isConstant()
           && "Dimvars snuck into .sign handling");
    size_t lhs_dim = lhs.dim->offset;
    size_t rhs_dim = rhs.dim->offset;

    // if this was bit[m] -> bit[n], then the phase type should be rev_qfunc[m]
    mlir::Type qbundle_type = handle.builder.getType<qwerty::QBundleType>(lhs_dim);
    qwerty::FunctionType mlir_stub_type =
        handle.builder.getType<qwerty::FunctionType>(
            handle.builder.getFunctionType({qbundle_type}, {qbundle_type}),
            /*reversible=*/true);

    // This will just be a stub function that calls the XOR embedding
    handle.builder.setInsertionPointToEnd(handle.module->getBody());
    auto stub_func = handle.builder.create<qwerty::FuncOp>(loc, stub_name, mlir_stub_type);
    // Sets insert point to end of this block
    mlir::Block *block = handle.builder.createBlock(&stub_func.getBody(), {}, {qbundle_type}, {loc});

    qwerty::BasisAttr basis = handle.builder.getAttr<qwerty::BasisAttr>(
        std::initializer_list<qwerty::BasisElemAttr>{
            handle.builder.getAttr<qwerty::BasisElemAttr>(
                handle.builder.getAttr<qwerty::BasisVectorListAttr>(
                    std::initializer_list<qwerty::BasisVectorAttr>{
                        handle.builder.getAttr<qwerty::BasisVectorAttr>(
                            qwerty::PrimitiveBasis::X, qwerty::Eigenstate::MINUS,
                            rhs_dim, /*hasPhase=*/false)}))});
    mlir::Value zero = handle.builder.create<qwerty::QBundlePrepOp>(
        loc, qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::PLUS, rhs_dim).getResult();
    mlir::Value minus = handle.builder.create<qwerty::QBundleInitOp>(
        loc, basis, mlir::ValueRange(), zero).getQbundleOut();
    mlir::ValueRange minus_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, minus).getQubits();

    mlir::Value qbundle_arg = block->getArgument(0);
    mlir::ValueRange arg_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, qbundle_arg).getQubits();
    llvm::SmallVector<mlir::Value> all_merged(arg_unpacked.begin(), arg_unpacked.end());
    all_merged.append(minus_unpacked.begin(), minus_unpacked.end());
    mlir::Value repacked = handle.builder.create<qwerty::QBundlePackOp>(loc, all_merged).getQbundle();

    mlir::ValueRange results = handle.builder.create<qwerty::CallOp>(loc, xor_embed, repacked).getResults();
    mlir::ValueRange results_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, results).getQubits();

    llvm::SmallVector<mlir::Value> output_qubits(results_unpacked.begin(), results_unpacked.begin()+lhs_dim);
    mlir::Value output = handle.builder.create<qwerty::QBundlePackOp>(loc, output_qubits).getQbundle();
    llvm::SmallVector<mlir::Value> uncompute_qubits(results_unpacked.begin()+lhs_dim, results_unpacked.end());
    mlir::Value to_uncompute = handle.builder.create<qwerty::QBundlePackOp>(loc, uncompute_qubits).getQbundle();

    mlir::Value to_discard = handle.builder.create<qwerty::QBundleDeinitOp>(
        loc, basis, mlir::ValueRange(), to_uncompute).getQbundleOut();
    handle.builder.create<qwerty::QBundleDiscardZeroOp>(loc, to_discard);
    handle.builder.create<qwerty::ReturnOp>(loc, output);

    // <=== Restore old insertion point
    handle.builder.restoreInsertionPoint(old_insertpt);

    // Has non-classical inputs
    stub_func.setPrivate();
    return stub_func;
}

qwerty::FuncOp constructInPlaceEmbedding(
        MlirHandle &handle,
        mlir::Location loc,
        qwerty::FuncOp fwd_xor_embed,
        qwerty::FuncOp rev_xor_embed,
        std::string stub_name,
        FuncType &classical_type) {
    // ===> Save insertion point
    auto old_insertpt = handle.builder.saveInsertionPoint();

    std::unique_ptr<Type> canon_classical_type = classical_type.canonicalize();
    FuncType &canon_func_type = dynamic_cast<FuncType &>(*canon_classical_type);
    BitType &lhs = dynamic_cast<BitType &>(*canon_func_type.lhs);
    [[maybe_unused]] BitType &rhs = dynamic_cast<BitType &>(*canon_func_type.rhs);
    assert(*lhs.dim == *rhs.dim
           && "In-place embedding requires same number of input and output bits");
    assert(lhs.dim->isConstant() && rhs.dim->isConstant()
           && "Dimvars snuck into .sign handling");
    size_t dim = lhs.dim->offset;

    // if this was bit[n] -> bit[n], then the phase type should be rev_qfunc[n]
    mlir::Type qbundle_type = handle.builder.getType<qwerty::QBundleType>(dim);
    qwerty::FunctionType mlir_stub_type =
        handle.builder.getType<qwerty::FunctionType>(
            handle.builder.getFunctionType({qbundle_type}, {qbundle_type}),
            /*reversible=*/true);

    // This will just be a stub function that calls the XOR embedding
    handle.builder.setInsertionPointToEnd(handle.module->getBody());
    auto stub_func = handle.builder.create<qwerty::FuncOp>(loc, stub_name, mlir_stub_type);
    // Sets insert point to end of this block
    mlir::Block *block = handle.builder.createBlock(&stub_func.getBody(), {}, {qbundle_type}, {loc});

    mlir::Value zeros = handle.builder.create<qwerty::QBundlePrepOp>(
            loc, qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::PLUS, dim).getResult();
    mlir::ValueRange zeros_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, zeros).getQubits();
    mlir::Value qbundle_arg = block->getArgument(0);
    mlir::ValueRange arg_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, qbundle_arg).getQubits();
    llvm::SmallVector<mlir::Value> all_merged(arg_unpacked.begin(), arg_unpacked.end());
    all_merged.append(zeros_unpacked.begin(), zeros_unpacked.end());
    mlir::Value repacked = handle.builder.create<qwerty::QBundlePackOp>(loc, all_merged).getQbundle();

    mlir::ValueRange results = handle.builder.create<qwerty::CallOp>(loc, fwd_xor_embed, repacked).getResults();
    mlir::ValueRange results_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, results).getQubits();

    llvm::SmallVector<mlir::Value> qubits(results_unpacked);
    for (size_t i = 0; i < dim; i++) {
        qcirc::Gate2QOp swap = handle.builder.create<qcirc::Gate2QOp>(
            loc, qcirc::Gate2Q::Swap, mlir::ValueRange(), qubits[i], qubits[i + dim]);
        qubits[i] = swap.getLeftResult();
        qubits[i + dim] = swap.getRightResult();
    }
    mlir::Value swapped_repacked = handle.builder.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();

    mlir::ValueRange rev = handle.builder.create<qwerty::CallOp>(loc, rev_xor_embed, swapped_repacked).getResults();
    assert(rev.size() == 1 && "XOR embedding should return 1 value");
    mlir::ValueRange rev_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, rev[0]).getQubits();

    llvm::SmallVector<mlir::Value> discard_qubits(rev_unpacked.begin()+dim, rev_unpacked.end());
    mlir::Value to_discard = handle.builder.create<qwerty::QBundlePackOp>(loc, discard_qubits).getQbundle();
    handle.builder.create<qwerty::QBundleDiscardZeroOp>(loc, to_discard);

    llvm::SmallVector<mlir::Value> output_qubits(rev_unpacked.begin(), rev_unpacked.begin()+dim);
    mlir::Value output = handle.builder.create<qwerty::QBundlePackOp>(loc, output_qubits).getQbundle();
    handle.builder.create<qwerty::ReturnOp>(loc, output);

    // <=== Restore old insertion point
    handle.builder.restoreInsertionPoint(old_insertpt);

    // Has non-classical inputs
    stub_func.setPrivate();
    return stub_func;
}
} // namespace

qwerty::FuncOp ClassicalKernel::getFuncOp(MlirHandle &handle, ClassicalKernel *operand, EmbeddingKind embedding) {
    assert(((embedding == EMBED_INPLACE) ^ !operand) && "operand should be supplied iff .inplace");

    if (!embedding_func_ops.count(embedding)) {
        assert(embedding_func_ops.count(EMBED_XOR) && "No XOR embedding compiled yet");

        qwerty::FuncOp new_embedding_func_op;
        mlir::Location loc = dbg->toMlirLoc(handle);
        qwerty::FuncOp xor_embed = embedding_func_ops.at(EMBED_XOR);
        std::string stub_name = getFuncOpName(embedding);
        FuncType &classical_type = dynamic_cast<FuncType &>(*type);

        switch (embedding) {
        case EMBED_SIGN:
            new_embedding_func_op = constructPhaseEmbedding(handle, loc, xor_embed, stub_name, classical_type);
            break;

        case EMBED_INPLACE:
            {
                qwerty::FuncOp rev_xor_embed = operand->getFuncOp(handle, nullptr, EMBED_XOR);
                new_embedding_func_op = constructInPlaceEmbedding(handle, loc, xor_embed, rev_xor_embed, stub_name, classical_type);
            }
            break;

        default:
            assert(0 && "Whoops, synthesis for embedding missing");
        }

        [[maybe_unused]] bool added =
            embedding_func_ops.insert({embedding, new_embedding_func_op}).second;
        assert(added && "Didn't add new embedding funcop, how?");
    }
    return embedding_func_ops.at(embedding);
}

void ClassicalKernel::compile(MlirHandle &handle) {
    Kernel::compile(handle);
    ClassicalNetlistVisitor visitor(capture_objs);
    walk(visitor);

    if (qwerty_debug) {
        mockturtle::write_dot(visitor.net, "mockturtle_net_" + name + ".dot");
    }

    TweedledumCircuit circ = TweedledumCircuit::fromNetlist(visitor.net);

    if (qwerty_debug) {
        circ.toFile(name);
    }

    // Tweedledum will generate a XOR embedding
    mlir::Location loc = dbg->toMlirLoc(handle);
    qwerty::FuncOp xor_embed = circ.toFuncOp(handle.builder, loc, *handle.module, getFuncOpName(EMBED_XOR));

    if (!xor_embed) {
        throw CompileException("Could not verify FuncOp generated from "
                               "tweedledum Circuit. See errors on stderr",
                               std::move(dbg->copy()));
    }

    [[maybe_unused]] bool added =
        embedding_func_ops.insert({EMBED_XOR, xor_embed}).second;
    assert(added && "XOR embedding already registered, how?");
}
