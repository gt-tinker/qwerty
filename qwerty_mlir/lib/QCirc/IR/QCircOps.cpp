//===- QCircOps.cpp - QCirc dialect ops --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include <unordered_set>
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "QCirc/IR/QCircOps.h"
#include "QCirc/IR/QCircDialect.h"
#include "QCirc/Transforms/QCircPasses.h"

#define GET_OP_CLASSES
#include "QCirc/IR/QCircOps.cpp.inc"

namespace {
// General Steps to check if a qubit IR operation is linear with many uses:
// - Same block does not have two different use of an qubit IR instruction
// - Different blocks are allowed for control flow cases which means they cannot be dominating each other
bool linearCheckForManyUses(mlir::Value &value) {
    std::unordered_set<mlir::Block *> blocks;
    mlir::DominanceInfo domInfo;

    for(auto user : value.getUsers()) {
        mlir::Block *block = user->getBlock();

        for(auto oblock : blocks) {
            if(oblock == block ||
               domInfo.dominates(block, oblock) ||
               domInfo.dominates(oblock, block)) {
                return false;
            }
        }

        blocks.insert(block);
    }

    return true;
}

// Simplify arrpack(arrunpack(arr)) to just arr. Similar to
// SimplifyPackUnpack in QwertyOps.cpp (which is for qbundles).
struct SimplifyArrayPackUnpack : public mlir::OpRewritePattern<qcirc::ArrayUnpackOp> {
    using OpRewritePattern<qcirc::ArrayUnpackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::ArrayUnpackOp unpack,
                                        mlir::PatternRewriter &rewriter) const override {
        mlir::Value arr = unpack.getArray();
        qcirc::ArrayPackOp pack = arr.getDefiningOp<qcirc::ArrayPackOp>();
        if (!pack) {
            return mlir::failure();
        }
        // If the qbundle is used multiple times, it must be used in different
        // branches of a conditional. In this case, we need to do this rewrite
        // on both paths of the conditional or we violate linearity. We can't
        // guarantee that safely here, so just don't bother simplifying.
        // TODO: What about an array of array of qubits? (Currently not
        //       possible to generate, though)
        if (llvm::isa<qcirc::QubitType>(
                llvm::cast<qcirc::ArrayType>(
                    arr.getType()).getElemType())
                && !pack.getArray().hasOneUse()) {
            return mlir::failure();
        }
        rewriter.replaceOp(unpack, pack.getElems());
        return mlir::success();
    }
};

// Simplify arrunpack(arrpack(e1,e2,...,en)) to e1,e2,...,en.
struct SimplifyArrayUnpackPack : public mlir::OpRewritePattern<qcirc::ArrayPackOp> {
    using OpRewritePattern<qcirc::ArrayPackOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::ArrayPackOp pack,
                                        mlir::PatternRewriter &rewriter) const override {
        assert(!pack.getElems().empty());

        qcirc::ArrayUnpackOp unpack =
            pack.getElems()[0].getDefiningOp<qcirc::ArrayUnpackOp>();
        if (!unpack) {
            return mlir::failure();
        }

        if (unpack.getElems() != pack.getElems()) {
            return mlir::failure();
        }

        // If the qubits being packed are used multiple times, they must be
        // used in different branches of a conditional. In this case, we need
        // to do this rewrite on both paths of the conditional or we violate
        // linearity. We can't guarantee that safely here, so just don't bother
        // simplifying.
        // TODO: What about an array of array of qubits? (Currently not
        //       possible to generate, though)
        for (mlir::Value elem : unpack.getElems()) {
            if (llvm::isa<qcirc::QubitType>(elem.getType())
                    && !elem.hasOneUse()) {
                return mlir::failure();
            }
        }

        mlir::Value arr = unpack.getArray();

        // Same reasoning as in SimplyArrayPackUnpack
        if (llvm::isa<qcirc::QubitType>(
                llvm::cast<qcirc::ArrayType>(
                    arr.getType()).getElemType())
                && !pack.getArray().hasOneUse()) {
            return mlir::failure();
        }

        rewriter.replaceOp(pack, arr);
        return mlir::success();
    }
};

// Consolidate chains of calc ops. For example, replace this IR
//     %1 = calc(%0 as %x) { %x + 1 }
//     %2 = calc(%1 as %y) { %y + 2 }
// with
//     %1 = calc(%0 as %x) { %x + 1 }
//     %2 = calc(%0 as %x) { (%x + 1) + 2 }
// If that was the last use of the first op, it will get folded away later.
struct ChainedCalcs : public mlir::OpRewritePattern<qcirc::CalcOp> {
    using OpRewritePattern<qcirc::CalcOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(qcirc::CalcOp calc,
                                        mlir::PatternRewriter &rewriter) const override {
        qcirc::CalcOp upstream;
        llvm::SmallVector<mlir::OpOperand *> upstream_edges;
        llvm::SmallVector<mlir::Value> new_inputs;
        llvm::SmallVector<mlir::Type> new_input_types;
        for (mlir::OpOperand &operand : calc->getOpOperands()) {
            mlir::Value input = operand.get();
            if (qcirc::CalcOp dad = input.getDefiningOp<qcirc::CalcOp>()) {
                if (!upstream) {
                    upstream = dad;
                    upstream_edges.push_back(&operand);
                } else if (upstream == dad) {
                    upstream_edges.push_back(&operand);
                } else {
                    // We're focusing on just one upstream CalcOp for now, so
                    // let's chill out and keep it around
                    new_inputs.push_back(input);
                    new_input_types.push_back(input.getType());
                }
            } else {
                // Keep this operand around, defining op is not a calc
                new_inputs.push_back(input);
                new_input_types.push_back(input.getType());
            }
        }
        if (!upstream) {
            return mlir::failure();
        }

        // Must return mlir::success() from this point onward

        size_t num_old_inputs = new_inputs.size();
        new_inputs.append(upstream.getInputs().begin(),
                          upstream.getInputs().end());
        new_input_types.append(upstream->operand_type_begin(),
                               upstream->operand_type_end());
        qcirc::CalcOp new_calc = rewriter.create<qcirc::CalcOp>(
            calc.getLoc(), calc->getResultTypes(), new_inputs);
        llvm::SmallVector<mlir::Location> arg_locs(new_input_types.size(), calc.getLoc());
        mlir::Block *new_calc_block = rewriter.createBlock(&new_calc.getRegion(), {}, new_input_types, arg_locs);

        rewriter.cloneRegionBefore(upstream.getRegion(), new_calc.getRegion(), new_calc.getRegion().end());
        mlir::Block *new_upstream_block = &new_calc.getRegion().back();
        qcirc::CalcYieldOp upstream_yield =
            llvm::cast<qcirc::CalcYieldOp>(new_upstream_block->getTerminator());
        llvm::SmallVector<mlir::Value> upstream_results(upstream_yield.getResults().begin(),
                                                        upstream_yield.getResults().end());
        rewriter.eraseOp(upstream_yield);

        llvm::SmallVector<mlir::Value> new_new_inputs(new_calc_block->args_begin() + num_old_inputs, new_calc_block->args_end());
        rewriter.mergeBlocks(new_upstream_block, new_calc_block, new_new_inputs);

        // Interleave original (untouched) arguments with new block arguments
        llvm::SmallVector<mlir::Value> new_old_inputs;
        size_t old_block_num_inputs = calc.getInputs().size();
        new_old_inputs.reserve(old_block_num_inputs);
        size_t upstream_edge_idx = 0;
        size_t new_block_arg_idx = 0;
        size_t old_block_arg_idx = 0;
        while (old_block_arg_idx < old_block_num_inputs) {
            mlir::OpOperand *operand;
            if (upstream_edge_idx < upstream_edges.size()
                    && old_block_arg_idx ==
                        (operand = upstream_edges[upstream_edge_idx])->getOperandNumber()) {
                mlir::OpResult res = llvm::cast<mlir::OpResult>(operand->get());
                new_old_inputs.push_back(upstream_results[res.getResultNumber()]);
                upstream_edge_idx++;
            } else {
                new_old_inputs.push_back(new_calc_block->getArgument(new_block_arg_idx));
                new_block_arg_idx++;
            }
            old_block_arg_idx++;
        }

        rewriter.inlineBlockBefore(&calc.getRegion().front(), new_calc_block, new_calc_block->end(), new_old_inputs);
        // Leave terminator alone, it's fine actually
        rewriter.replaceOp(calc, new_calc);
        return mlir::success();
    }
};
} // namespace


namespace qcirc {

mlir::Value stationaryF64Negate(mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value theta) {
    qcirc::CalcOp calc = rewriter.create<qcirc::CalcOp>(loc, rewriter.getF64Type(), theta);
    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        // Sets insertion point to end of this block
        mlir::Block *calc_block = rewriter.createBlock(&calc.getRegion(), {}, rewriter.getF64Type(), {loc});
        assert(calc_block->getNumArguments() == 1);
        mlir::Value old_theta = calc_block->getArgument(0);
        mlir::Value neg_theta = rewriter.create<mlir::arith::NegFOp>(loc, old_theta).getResult();
        rewriter.create<qcirc::CalcYieldOp>(loc, neg_theta);
    }
    mlir::ValueRange calc_results = calc.getResults();
    assert(calc_results.size() == 1);
    return calc_results[0];
}

// Yanked from AdjointOp below
mlir::ParseResult CalcOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::OpAsmParser::Argument> args;
    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::UnresolvedOperand operand;
                mlir::OpAsmParser::Argument arg;
                if (parser.parseOperand(operand)
                        || parser.parseKeyword("as")
                        || parser.parseArgument(arg)) {
                    return mlir::failure();
                }
                operands.push_back(operand);
                args.push_back(arg);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    mlir::FunctionType func_type;
    if (parser.parseColonType(func_type)
        || parser.resolveOperands(operands, func_type.getInputs(), parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }

    result.addTypes(func_type.getResults());

    for (size_t i = 0; i < func_type.getInputs().size(); i++) {
        args[i].type = func_type.getInputs()[i];
    }

    // Introduce the body region and parse it.
    mlir::Region *body = result.addRegion();
    if (parser.parseRegion(*body, args) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    return mlir::success();
}

// Yanked from AdjointOp below
void CalcOp::print(mlir::OpAsmPrinter &p) {
    p << '(';
    llvm::interleaveComma(llvm::zip_equal(getInputs(), getRegion().getArguments()), p.getStream(), [&](auto pair) {
        auto [input, arg] = pair;
        p.printOperand(input);
        p << " as ";
        p.printRegionArgument(arg, /*argAttrs=*/{}, /*omitType=*/true);
    });
    p << ") : ";
    p << mlir::FunctionType::get(getContext(), getOperandTypes(), getResultTypes());

    p << ' ';
    p.printRegion(getRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);

    p.printOptionalAttrDict((*this)->getAttrs());
}

// Based on qwerty.pred
void CalcOp::getSuccessorRegions(
        mlir::RegionBranchPoint point, llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
    // Branch from parent region into this region
    if (point.isParent()) {
        regions.push_back(mlir::RegionSuccessor(&getRegion()));
        return;
    }
    // Branch back into parent region
    regions.push_back(mlir::RegionSuccessor(getResults()));
}

void CalcOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         mlir::MLIRContext *context) {
    results.add<ChainedCalcs>(context);
}

mlir::LogicalResult CalcOp::fold(
        FoldAdaptor adaptor,
        llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
    mlir::Region &body = adaptor.getRegion();
    assert(body.hasOneBlock());
    mlir::Block &block = body.front();
    CalcYieldOp yield = llvm::cast<CalcYieldOp>(block.getTerminator());
    llvm::SmallVector<mlir::OpFoldResult> floats;
    for (mlir::Value ret : yield.getResults()) {
        mlir::FloatAttr attr;
        if (!mlir::matchPattern(ret, mlir::m_Constant(&attr))) {
            return mlir::failure();
        }
        floats.emplace_back(attr);
    }
    results = floats;
    return mlir::success();
}

// Based on both PredOp in the qwerty dialect and scf.execute_region
mlir::ParseResult AdjointOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<mlir::OpAsmParser::Argument> args;
    if (parser.parseCommaSeparatedList(
            mlir::OpAsmParser::Delimiter::Paren, [&]() -> mlir::ParseResult {
                mlir::OpAsmParser::UnresolvedOperand operand;
                mlir::OpAsmParser::Argument arg;
                if (parser.parseOperand(operand)
                        || parser.parseKeyword("as")
                        || parser.parseArgument(arg)) {
                    return mlir::failure();
                }
                operands.push_back(operand);
                args.push_back(arg);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    mlir::FunctionType func_type;
    if (parser.parseColonType(func_type)
        || parser.resolveOperands(operands, func_type.getInputs(), parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }

    result.addTypes(func_type.getResults());

    for (size_t i = 0; i < func_type.getInputs().size(); i++) {
        args[i].type = func_type.getInputs()[i];
    }

    // Introduce the body region and parse it.
    mlir::Region *body = result.addRegion();
    if (parser.parseRegion(*body, args) ||
        parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    return mlir::success();
}

void AdjointOp::print(mlir::OpAsmPrinter &p) {
    p << '(';
    llvm::interleaveComma(llvm::zip_equal(getInputs(), getRegion().getArguments()), p.getStream(), [&](auto pair) {
        auto [input, arg] = pair;
        p.printOperand(input);
        p << " as ";
        p.printRegionArgument(arg, /*argAttrs=*/{}, /*omitType=*/true);
    });
    p << ") : ";
    p << mlir::FunctionType::get(getContext(), getOperandTypes(), getResultTypes());

    p << ' ';
    p.printRegion(getRegion(),
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);

    p.printOptionalAttrDict((*this)->getAttrs());
}

void AdjointOp::getSuccessorRegions(
        mlir::RegionBranchPoint point, llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
    // Branch from parent region into this region
    if (point.isParent()) {
        regions.push_back(mlir::RegionSuccessor(&getRegion()));
        return;
    }
    // Branch back into parent region
    regions.push_back(mlir::RegionSuccessor(getResults()));
}

mlir::LogicalResult AdjointOp::verify() {
    if (getInputs().size() != getRegion().getNumArguments()) {
        return this->emitOpError("AdjointOp must have same number of inputs as "
                                 "number of region arguments");
    }

    for (mlir::BlockArgument arg : getRegion().getArguments()) {
        if (!(arg.hasOneUse() || linearCheckForManyUses(arg))) {
            return this->emitOpError("AdjointOp: Argument of region (a qubit) "
                                     "is not linear");
        }
    }

    auto results = getResults();

    for (auto indexedResult : llvm::enumerate(results)) {
        mlir::Value result = indexedResult.value();
        if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
            return this->emitOpError("AdjointOp: ")
                << "Result(" << indexedResult.index()
                << ") qubit is not linear with this IR instruction (gate)";
        }
    }

    return mlir::success();
}

// TODO: de-bloat these gate print/parse routines with some macros or something.
//       currently i am just doing it this way so that i can read the IR ASAP
//       (the default printing is pretty unreadable)

mlir::ParseResult Gate1QOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::StringRef gateName;
    std::optional<Gate1Q> gate;
    mlir::OpAsmParser::UnresolvedOperand qubit;
    if (parser.parseOperandList(operands, mlir::AsmParser::Delimiter::Square)
            || parser.parseColon()
            || parser.parseKeyword(&gateName)
            || !(gate = symbolizeGate1Q(gateName))
            || parser.parseOperand(qubit)) {
        return mlir::failure();
    }
    operands.push_back(qubit);

    mlir::FunctionType funcType;
    if (parser.parseColonType(funcType)
            || parser.resolveOperands(operands, funcType.getInputs(), parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }
    result.addTypes(funcType.getResults());
    result.addAttribute(getGateAttrStrName(), Gate1QAttr::get(parser.getContext(), gate.value()));
    return mlir::success();
}

void Gate1QOp::print(mlir::OpAsmPrinter &p) {
    p << '[';
    if (!getControls().empty()) {
        p.printOperands(getControls());
    }
    p << "]:";
    p << stringifyGate1Q(getGate());
    p << ' ';
    p.printOperand(getQubit());
    p << " : ";
    p << mlir::FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

mlir::LogicalResult Gate1QOp::verify() {
    auto result = getResult();
    auto ctrlResults = getControlResults();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
        return this->emitOpError("Gate1QOp: ")
            << "Result qubit is not linear with this IR instruction (gate)";
    }

    for (auto indexedResult : llvm::enumerate(ctrlResults)) {
        mlir::Value ctrlResult = indexedResult.value();

        if (!(ctrlResult.hasOneUse() || linearCheckForManyUses(ctrlResult))) {
            return this->emitOpError("Gate1QOp: ")
                << "Control Result(" << indexedResult.index()
                << ") qubit is not linear with this IR instruction (gate)";
        }
    }

    return mlir::success();
}

void Gate1QOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    Gate1QOpAdaptor adaptor(newInputs);

    Gate1Q adj_gate;
    switch (getGate()) {
    case Gate1Q::X:
    case Gate1Q::Y:
    case Gate1Q::Z:
    case Gate1Q::H:
        adj_gate = getGate(); // Hermitian
        break;
    case Gate1Q::S:
        adj_gate = Gate1Q::Sdg;
        break;
    case Gate1Q::Sdg:
        adj_gate = Gate1Q::S;
        break;
    case Gate1Q::Sx:
        adj_gate = Gate1Q::Sxdg;
        break;
    case Gate1Q::Sxdg:
        adj_gate = Gate1Q::Sx;
        break;
    case Gate1Q::T:
        adj_gate = Gate1Q::Tdg;
        break;
    case Gate1Q::Tdg:
        adj_gate = Gate1Q::T;
        break;
    default:
        assert(0 && "Missing Gate1Q in Gate1Q::buildAdjoint()");
        adj_gate = (Gate1Q)-1;
    }

    Gate1QOp adj = rewriter.create<Gate1QOp>(
        getLoc(), adj_gate, adaptor.getControls(), adaptor.getQubit());
    newOutputs.clear();
    newOutputs.append(adj->result_begin(), adj->result_end());
}

void Gate1QOp::buildControlled(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange controlsIn,
        llvm::SmallVectorImpl<mlir::Value> &controlsOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    Gate1QOpAdaptor adaptor(newInputs);
    llvm::SmallVector<mlir::Value> ctrls(controlsIn.begin(), controlsIn.end());
    ctrls.append(adaptor.getControls().begin(), adaptor.getControls().end());

    Gate1QOp gate = rewriter.create<Gate1QOp>(
        getLoc(), getGate(), ctrls, adaptor.getQubit());

    size_t n_new_controls = controlsIn.size();
    controlsOut.clear();
    controlsOut.append(gate.getControlResults().begin(),
                       gate.getControlResults().begin() + n_new_controls);
    newOutputs.clear();
    newOutputs.append(gate->result_begin() + n_new_controls,
                      gate->result_end());
}

mlir::ParseResult Gate1Q1POp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::StringRef gateName;
    std::optional<Gate1Q1P> gate;
    mlir::OpAsmParser::UnresolvedOperand param, qubit;
    if (parser.parseOperandList(operands, mlir::AsmParser::Delimiter::Square)
            || parser.parseColon()
            || parser.parseKeyword(&gateName)
            || !(gate = symbolizeGate1Q1P(gateName))
            || parser.parseLParen()
            || parser.parseOperand(param)
            || parser.parseRParen()
            || parser.parseOperand(qubit)) {
        return mlir::failure();
    }
    operands.insert(operands.begin(), param);
    operands.push_back(qubit);

    mlir::FunctionType funcType;
    if (parser.parseColonType(funcType)
            || parser.resolveOperands(operands, funcType.getInputs(), parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }
    result.addTypes(funcType.getResults());
    result.addAttribute(getGateAttrStrName(), Gate1Q1PAttr::get(parser.getContext(), gate.value()));
    return mlir::success();
}

void Gate1Q1POp::print(mlir::OpAsmPrinter &p) {
    p << '[';
    if (!getControls().empty()) {
        p.printOperands(getControls());
    }
    p << "]:";
    p << stringifyGate1Q1P(getGate());
    p << '(';
    p.printOperand(getParam());
    p << ") ";
    p.printOperand(getQubit());
    p << " : ";
    p << mlir::FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

mlir::LogicalResult Gate1Q1POp::verify() {
    auto result = getResult();
    auto ctrlResults = getControlResults();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
        return this->emitOpError("Gate1Q1POp: ")
            << "Result qubit is not linear with this IR instruction (gate)";
    }

    for (auto indexedResult : llvm::enumerate(ctrlResults)) {
        mlir::Value ctrlResult = indexedResult.value();

        if (!(ctrlResult.hasOneUse() || linearCheckForManyUses(ctrlResult))) {
            return this->emitOpError("Gate1Q1POp: ")
                << "Control Result(" << indexedResult.index()
                << ") qubit is not linear with this IR instruction (gate)";
        }
    }

    return mlir::success();
}

void Gate1Q1POp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert((getGate() == Gate1Q1P::P
            || getGate() == Gate1Q1P::Rx
            || getGate() == Gate1Q1P::Ry
            || getGate() == Gate1Q1P::Rz)
           && "Missing Gate1Q1P in Gate1Q1POp::buildAdjoint()");

    Gate1Q1POpAdaptor adaptor(newInputs);
    mlir::Value neg_theta = stationaryF64Negate(rewriter, getLoc(),
                                                adaptor.getParam());

    Gate1Q1POp adj = rewriter.create<Gate1Q1POp>(
        getLoc(), getGate(), neg_theta, adaptor.getControls(), adaptor.getQubit());
    newOutputs.clear();
    newOutputs.append(adj->result_begin(), adj->result_end());
}

void Gate1Q1POp::buildControlled(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange controlsIn,
        llvm::SmallVectorImpl<mlir::Value> &controlsOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    Gate1Q1POpAdaptor adaptor(newInputs);
    llvm::SmallVector<mlir::Value> ctrls(controlsIn.begin(), controlsIn.end());
    ctrls.append(adaptor.getControls().begin(), adaptor.getControls().end());

    Gate1Q1POp gate = rewriter.create<Gate1Q1POp>(
        getLoc(), getGate(), adaptor.getParam(), ctrls, adaptor.getQubit());

    size_t n_new_controls = controlsIn.size();
    controlsOut.clear();
    controlsOut.append(gate.getControlResults().begin(),
                       gate.getControlResults().begin() + n_new_controls);
    newOutputs.clear();
    newOutputs.append(gate->result_begin() + n_new_controls,
                      gate->result_end());
}

mlir::ParseResult Gate1Q3POp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::StringRef gateName;
    std::optional<Gate1Q3P> gate;
    mlir::OpAsmParser::UnresolvedOperand param1, param2, param3, qubit;
    if (parser.parseOperandList(operands, mlir::AsmParser::Delimiter::Square)
            || parser.parseColon()
            || parser.parseKeyword(&gateName)
            || !(gate = symbolizeGate1Q3P(gateName))
            || parser.parseLParen()
            || parser.parseOperand(param1)
            || parser.parseComma()
            || parser.parseOperand(param2)
            || parser.parseComma()
            || parser.parseOperand(param3)
            || parser.parseRParen()
            || parser.parseOperand(qubit)) {
        return mlir::failure();
    }
    operands.insert(operands.begin(), param1);
    operands.insert(operands.begin()+1, param2);
    operands.insert(operands.begin()+2, param3);
    operands.push_back(qubit);

    mlir::FunctionType funcType;
    if (parser.parseColonType(funcType)
            || parser.resolveOperands(operands, funcType.getInputs(), parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }
    result.addTypes(funcType.getResults());
    result.addAttribute(getGateAttrStrName(), Gate1Q3PAttr::get(parser.getContext(), gate.value()));
    return mlir::success();
}

void Gate1Q3POp::print(mlir::OpAsmPrinter &p) {
    p << '[';
    if (!getControls().empty()) {
        p.printOperands(getControls());
    }
    p << "]:";
    p << stringifyGate1Q3P(getGate());
    p << '(';
    p.printOperand(getFirstParam());
    p << ", ";
    p.printOperand(getSecondParam());
    p << ", ";
    p.printOperand(getThirdParam());
    p << ") ";
    p.printOperand(getQubit());
    p << " : ";
    p << mlir::FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

mlir::LogicalResult Gate1Q3POp::verify() {
    auto result = getResult();
    auto ctrlResults = getControlResults();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
        return this->emitOpError("Gate1Q3POp: ")
            << "Result qubit is not linear with this IR instruction (gate)";
    }

    for (auto indexedResult : llvm::enumerate(ctrlResults)) {
        mlir::Value ctrlResult = indexedResult.value();

        if (!(ctrlResult.hasOneUse() || linearCheckForManyUses(ctrlResult))) {
            return this->emitOpError("Gate1Q3POp: ")
                << "Control Result(" << indexedResult.index()
                << ") qubit is not linear with this IR instruction (gate)";
        }
    }

    return mlir::success();
}

void Gate1Q3POp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(getGate() == Gate1Q3P::U
           && "Missing Gate1Q1P in Gate1Q3POp::buildAdjoint()");

    Gate1Q3POpAdaptor adaptor(newInputs);

    mlir::Value theta = adaptor.getFirstParam();
    mlir::Value phi = adaptor.getSecondParam();
    mlir::Value lambda = adaptor.getThirdParam();

    mlir::Value neg_theta = stationaryF64Negate(rewriter, getLoc(), theta);
    mlir::Value neg_phi = stationaryF64Negate(rewriter, getLoc(), phi);
    mlir::Value neg_lambda = stationaryF64Negate(rewriter, getLoc(), lambda);

    // U(theta, phi, lambda)^dagger = U(-theta, -lambda, -phi)
    // Source: i proved it on piece of paper
    Gate1Q3POp adj = rewriter.create<Gate1Q3POp>(
        getLoc(), getGate(), neg_theta, neg_lambda, neg_phi,
        adaptor.getControls(), adaptor.getQubit());
    newOutputs.clear();
    newOutputs.append(adj->result_begin(), adj->result_end());
}

void Gate1Q3POp::buildControlled(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange controlsIn,
        llvm::SmallVectorImpl<mlir::Value> &controlsOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    Gate1Q3POpAdaptor adaptor(newInputs);
    llvm::SmallVector<mlir::Value> ctrls(controlsIn.begin(), controlsIn.end());
    ctrls.append(adaptor.getControls().begin(), adaptor.getControls().end());

    Gate1Q3POp gate = rewriter.create<Gate1Q3POp>(
        getLoc(), getGate(), adaptor.getFirstParam(), adaptor.getSecondParam(),
        adaptor.getThirdParam(), ctrls, adaptor.getQubit());

    size_t n_new_controls = controlsIn.size();
    controlsOut.clear();
    controlsOut.append(gate.getControlResults().begin(),
                       gate.getControlResults().begin() + n_new_controls);
    newOutputs.clear();
    newOutputs.append(gate->result_begin() + n_new_controls,
                      gate->result_end());
}

mlir::ParseResult Gate2QOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> operands;
    llvm::StringRef gateName;
    std::optional<Gate2Q> gate;
    mlir::OpAsmParser::UnresolvedOperand leftQubit, rightQubit;
    if (parser.parseOperandList(operands, mlir::AsmParser::Delimiter::Square)
            || parser.parseColon()
            || parser.parseKeyword(&gateName)
            || !(gate = symbolizeGate2Q(gateName))
            || parser.parseOperand(leftQubit)
            || parser.parseComma()
            || parser.parseOperand(rightQubit)) {
        return mlir::failure();
    }
    operands.push_back(leftQubit);
    operands.push_back(rightQubit);

    mlir::FunctionType funcType;
    if (parser.parseColonType(funcType)
            || parser.resolveOperands(operands, funcType.getInputs(), parser.getCurrentLocation(), result.operands)) {
        return mlir::failure();
    }
    result.addTypes(funcType.getResults());
    result.addAttribute(getGateAttrStrName(), Gate2QAttr::get(parser.getContext(), gate.value()));
    return mlir::success();
}

void Gate2QOp::print(mlir::OpAsmPrinter &p) {
    p << '[';
    if (!getControls().empty()) {
        p.printOperands(getControls());
    }
    p << "]:";
    p << stringifyGate2Q(getGate());
    p << ' ';
    p.printOperand(getLeftQubit());
    p << ", ";
    p.printOperand(getRightQubit());
    p << " : ";
    p << mlir::FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

mlir::LogicalResult Gate2QOp::verify() {
    auto lResult = getLeftResult();
    auto rResult = getRightResult();
    auto ctrlResults = getControlResults();

    if (!(lResult.hasOneUse() || linearCheckForManyUses(lResult))) {
        return this->emitOpError("Gate2QOp: ")
                << "Left result qubit is not linear with this IR instruction (gate)";
    }

    if (!(rResult.hasOneUse() || linearCheckForManyUses(rResult))) {
        return this->emitOpError("Gate2QOp: ")
                << "Right result qubit is not linear with this IR instruction (gate)";
    }

    for (auto indexedResult : llvm::enumerate(ctrlResults)) {
        mlir::Value ctrlResult = indexedResult.value();

        if (!(ctrlResult.hasOneUse() || linearCheckForManyUses(ctrlResult))) {
            return this->emitOpError("Gate2QOp: ")
                    << "Control Result(" << indexedResult.index()
                    << ") qubit is not linear with this IR instruction (gate)";
        }
    }

    return mlir::success();
}

void Gate2QOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(getGate() == Gate2Q::Swap
           && "Missing Gate2Q in Gate2QOp::buildAdjoint()");

    Gate2QOpAdaptor adaptor(newInputs);

    // SWAP is Hermitian, so recreate ourself
    Gate2QOp adj = rewriter.create<Gate2QOp>(
        getLoc(), getGate(), adaptor.getControls(),
        adaptor.getLeftQubit(), adaptor.getRightQubit());
    newOutputs.clear();
    newOutputs.append(adj->result_begin(), adj->result_end());
}

void Gate2QOp::buildControlled(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange controlsIn,
        llvm::SmallVectorImpl<mlir::Value> &controlsOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    Gate2QOpAdaptor adaptor(newInputs);
    llvm::SmallVector<mlir::Value> ctrls(controlsIn.begin(), controlsIn.end());
    ctrls.append(adaptor.getControls().begin(), adaptor.getControls().end());

    Gate2QOp gate = rewriter.create<Gate2QOp>(
        getLoc(), getGate(), ctrls,
        adaptor.getLeftQubit(), adaptor.getRightQubit());

    size_t n_new_controls = controlsIn.size();
    controlsOut.clear();
    controlsOut.append(gate.getControlResults().begin(),
                       gate.getControlResults().begin() + n_new_controls);
    newOutputs.clear();
    newOutputs.append(gate->result_begin() + n_new_controls,
                      gate->result_end());
}

mlir::LogicalResult QallocOp::verify() {
    auto result = getResult();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
       return this->emitOpError("QallocOp: ")
               << "Result qubit is not linear with this IR instruction (gate)";
    }

    return mlir::success();
}

unsigned QallocOp::getNumOperandsOfAdjoint() {
    return 1; // Will become a qfree
}

void QallocOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QfreeZeroOpAdaptor adaptor(newInputs);
    rewriter.create<QfreeZeroOp>(getLoc(), adaptor.getQubit());
    newOutputs.clear();
}

void QallocOp::buildControlled(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange controlsIn,
        llvm::SmallVectorImpl<mlir::Value> &controlsOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QallocOp qalloc = rewriter.create<QallocOp>(getLoc());
    controlsOut.clear();
    controlsOut.append(controlsIn.begin(), controlsIn.end());
    newOutputs.clear();
    newOutputs.push_back(qalloc.getResult());
}

unsigned QfreeZeroOp::getNumOperandsOfAdjoint() {
    return 0; // Will become a qalloc
}

void QfreeZeroOp::buildAdjoint(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    assert(newInputs.empty());
    QallocOp qalloc = rewriter.create<QallocOp>(getLoc());
    newOutputs.clear();
    newOutputs.push_back(qalloc.getResult());
}

void QfreeZeroOp::buildControlled(
        mlir::RewriterBase &rewriter,
        mlir::ValueRange controlsIn,
        llvm::SmallVectorImpl<mlir::Value> &controlsOut,
        mlir::ValueRange newInputs,
        llvm::SmallVectorImpl<mlir::Value> &newOutputs) {
    QfreeZeroOpAdaptor adaptor(newInputs);
    rewriter.create<QfreeOp>(getLoc(), adaptor.getQubit());
    controlsOut.clear();
    controlsOut.append(controlsIn.begin(), controlsIn.end());
    newOutputs.clear();
}

mlir::LogicalResult QubitIndexOp::verify() {
    auto result = getResult();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
       return this->emitOpError("Result qubit is not linear with this IR "
                                "instruction");
    }

    return mlir::success();
}

mlir::LogicalResult MeasureOp::verify() {
    auto result = getQubitResult();

    if (!(result.hasOneUse() || linearCheckForManyUses(result))) {
        return this->emitOpError("MeasureOp: ")
                << "Result qubit is not linear with this IR instruction (gate)";
    }

    return mlir::success();
}

#define INFER_RETURN_TYPES_FOR(name, n_qubits) \
    mlir::LogicalResult name::inferReturnTypes( \
            mlir::MLIRContext *ctx, \
            std::optional<mlir::Location> loc, \
            name::Adaptor adaptor, \
            llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) { \
        size_t n_controls = adaptor.getControls().size(); \
        inferredReturnTypes.append(n_controls+n_qubits, QubitType::get(ctx)); \
        return mlir::success(); \
    }

INFER_RETURN_TYPES_FOR(Gate1QOp, 1)
INFER_RETURN_TYPES_FOR(Gate1Q1POp, 1)
INFER_RETURN_TYPES_FOR(Gate1Q3POp, 1)
INFER_RETURN_TYPES_FOR(Gate2QOp, 2)

#undef INFER_RETURN_TYPES_FOR

mlir::LogicalResult ArrayPackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        ArrayPackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    mlir::Type elem_type;
    for (mlir::Value elem : adaptor.getElems()) {
        mlir::Type this_type = elem.getType();
        if (!elem_type) {
            elem_type = this_type;
        } else if (elem_type != this_type) {
            return mlir::failure();
        }
    }
    size_t n_elems = adaptor.getElems().size();
    ArrayType ret_type = ArrayType::get(ctx, elem_type, n_elems);
    inferredReturnTypes.insert(inferredReturnTypes.end(), ret_type);
    return mlir::success();
}

mlir::LogicalResult ArrayPackOp::verify() {
    if (getElems().empty()) {
        return emitOpError("Cannot construct empty array");
    }

    mlir::Type elem_type;
    for (mlir::Value elem : getElems()) {
        mlir::Type this_type = elem.getType();
        if (!elem_type) {
            elem_type = this_type;
        } else if (elem_type != this_type) {
            return emitOpError("Mismatch in types passed: ")
                   << elem_type << " != " << this_type;
        }
    }

    if (llvm::isa<QubitType>(elem_type)) {
        mlir::Value arr = getArray();
        if (!(arr.hasOneUse() || linearCheckForManyUses(arr))) {
            return this->emitOpError("Resulting qubit array is not linear "
                                     "with this IR instruction");
        }
    }

    return mlir::success();
}

void ArrayPackOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              mlir::MLIRContext *context) {
    results.add<SimplifyArrayUnpackPack>(context);
}

mlir::LogicalResult ArrayUnpackOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        ArrayUnpackOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    ArrayType arr_type = llvm::cast<ArrayType>(adaptor.getArray().getType());
    mlir::Type elem_type = arr_type.getElemType();
    size_t n_elems = arr_type.getDim();
    if (!n_elems) {
        return mlir::failure();
    }
    inferredReturnTypes.insert(inferredReturnTypes.end(), n_elems, elem_type);
    return mlir::success();
}

mlir::LogicalResult ArrayUnpackOp::verify() {
    for (mlir::Value elem : getElems()) {
        if (llvm::isa<QubitType>(elem.getType())) {
            if (!(elem.hasOneUse() || linearCheckForManyUses(elem))) {
                return this->emitOpError("Unpacked qubit is not linear "
                                         "with this IR instruction");
            }
        }
    }

    return mlir::success();
}

void ArrayUnpackOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                                    mlir::MLIRContext *context) {
    results.add<SimplifyArrayPackUnpack>(context);
}

void CallableMetadataOp::print(mlir::OpAsmPrinter &p) {
    p << ' ';
    if (mlir::StringAttr sym_vis =
            getOperation()->getAttrOfType<mlir::StringAttr>(
                mlir::SymbolTable::getVisibilityAttrName())) {
        p << sym_vis.getValue() << ' ';
    }
    p.printSymbolName(getSymName());
    p << " captures [";
    llvm::interleaveComma(getCaptureTypeRange(), p, [&](mlir::Type type) {
        p.printType(type);
    });
    p << "] specs [";
    llvm::interleaveComma(getSpecsRange(), p, [&](FuncSpecAttr spec) {
        p.printStrippedAttrOrType(spec);
    });
    p << ']';
}

mlir::ParseResult CallableMetadataOp::parse(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
    (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

    mlir::StringAttr sym_name;
    if (parser.parseSymbolName(sym_name, mlir::SymbolTable::getSymbolAttrName(),
                               result.attributes)) {
        return mlir::failure();
    }
    if (parser.parseKeyword("captures")) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Type> capture_types;
    if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square, [&]() {
                mlir::Type ty;
                if (parser.parseType(ty)) {
                    return mlir::failure();
                }
                capture_types.push_back(ty);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    if (parser.parseKeyword("specs")) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Attribute> func_specs;
    if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square, [&]() {
                FuncSpecAttr spec;
                if (parser.parseCustomAttributeWithFallback(spec)) {
                    return mlir::failure();
                }
                func_specs.push_back(spec);
                return mlir::success();
            })) {
        return mlir::failure();
    }

    auto &props = result.getOrAddProperties<CallableMetadataOp::Properties>();
    props.sym_name = sym_name;
    props.capture_types = parser.getBuilder().getTypeArrayAttr(capture_types);
    props.specs = parser.getBuilder().getArrayAttr(func_specs);
    return mlir::success();
}

mlir::LogicalResult CallableMetadataOp::verify() {
    if (getSpecs().empty()) {
        return emitOpError("Cannot have empty list of specializations");
    }
    return mlir::success();
}

mlir::LogicalResult CallableCreateOp::verifySymbolUses(
        mlir::SymbolTableCollection &symbolTable) {
    CallableMetadataOp metadata;
    if (!(metadata = symbolTable.lookupNearestSymbolFrom<CallableMetadataOp>(
            getOperation(), getMetadataAttr()))) {
        return emitOpError("no callable_metadata op named ") << getMetadata();
    }

    llvm::SmallVector<mlir::Type> metadata_capture_types_vec(
        metadata.getCaptureTypeRange());
    mlir::TypeRange metadata_capture_types(metadata_capture_types_vec);
    mlir::TypeRange create_capture_types = getCaptures().getTypes();

    if (metadata_capture_types != create_capture_types) {
        return emitOpError("Mismatch in capture types");
    }

    return mlir::success();
}

mlir::LogicalResult CallableInvokeOp::inferReturnTypes(
        mlir::MLIRContext *ctx,
        std::optional<mlir::Location> loc,
        CallableInvokeOp::Adaptor adaptor,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    mlir::FunctionType inner_func_ty =
        llvm::cast<CallableType>(adaptor.getCallable().getType())
            .getFunctionType();
    inferredReturnTypes.append(inner_func_ty.getResults().begin(),
                               inner_func_ty.getResults().end());
    return mlir::success();
}

} // namespace qcirc
