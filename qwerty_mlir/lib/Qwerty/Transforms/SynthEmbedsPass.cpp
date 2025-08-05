#include <queue>

#include "mlir/IR/SymbolTable.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/Transforms/QwertyPasses.h"
#include "CCirc/IR/CCircOps.h"
#include "tweedledum.hpp"
#include "PassDetail.h"

// This is a pass that converts all qwerty.embed_* ops that reference classical
// circuits into qwerty.func_const ops referencing qwerty.funcs that contain
// quantum circuits.

namespace {

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
            qwerty::FuncOp xor_func;

            for (mlir::Operation *user : users) {
                if (qwerty::EmbedXorOp embed_xor = llvm::dyn_cast<qwerty::EmbedXorOp>(user)) {
                    if (!xor_func) {
                        qwerty::FunctionType func_ty = qwerty::EmbedXorOp::getQwertyFuncTypeOf(circ);
                        std::string embed_func_name = circ.getSymName().str() + "__xor";

                        rewriter.setInsertionPointAfter(circ);
                        xor_func = rewriter.create<qwerty::FuncOp>(loc, embed_func_name, func_ty);
                        xor_func.setPrivate();
                        mlir::Block *block = rewriter.createBlock(
                            &xor_func.getBody(), {}, func_ty.getFunctionType().getInputs(), {loc});

                        assert(block->getNumArguments() == 1
                               && "Wrong number of arguments for embed block");
                        qwerty::QBundleUnpackOp unpack = rewriter.create<qwerty::QBundleUnpackOp>(
                            loc, block->getArgument(0));
                        llvm::SmallVector<mlir::Value> qubits(unpack.getQubits());

                        TweedledumCircuit::fromCCirc(circ).toQCircInline(rewriter, loc, qubits, 0);

                        mlir::Value packed =
                            rewriter.create<qwerty::QBundlePackOp>(loc, qubits).getQbundle();
                        rewriter.create<qwerty::ReturnOp>(loc, packed);
                    }

                    rewriter.setInsertionPoint(embed_xor);
                    rewriter.replaceOpWithNewOp<qwerty::FuncConstOp>(embed_xor, xor_func);
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
