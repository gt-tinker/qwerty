#include "mockturtle/networks/xag.hpp"
#include "mockturtle/generators/modular_arithmetic.hpp"

#include "defs.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"

namespace {
// Overwrites n
template<class Ntk>
std::vector<mockturtle::signal<Ntk>> bit_rotate(
        Ntk& ntk, std::vector<mockturtle::signal<Ntk>> &n,
        std::vector<mockturtle::signal<Ntk>> &k, bool left) {
    // Calculate the number of bits needed to specify k in theory. This is
    // ceil(log2(n))
    size_t num_bits = n.size();
    assert(num_bits && "n is zero bits???");
    // Round number of bits up to a power of 2
    if (bits_popcount(num_bits) > 1) {
        num_bits = 1ULL << bits_needed(num_bits);
    }
    // Then find the index of the most significant 1, which should be log2 of
    // the number
    size_t k_bits = bits_needed(num_bits)-1;
    assert(k_bits && "No bits needed for k? How?");

    // Barrel shifter:
    // https://www.d.umn.edu/~gshute/logic/barrel-shifter.html
    std::vector<mockturtle::signal<Ntk>> result(n.begin(), n.end());
    for (size_t i = 0; i < std::min(k_bits, k.size()); i++) {
        std::vector<mockturtle::signal<Ntk>> unshifted(result.begin(), result.end());
        size_t middle_idx = 1ULL << i;
        if (!left) {
            // A right rotate by k is a left rotate by n_bits-k
            middle_idx = n.size() - middle_idx;
        }
        std::rotate(result.begin(), result.begin() + middle_idx, result.end());
        mockturtle::mux_inplace(ntk, k[k.size() - i - 1], result, unshifted);
    }
    return result;
}
} // namespace

void ClassicalNetlistVisitor::init(ASTNode &root) {
    ClassicalKernel &kernel = dynamic_cast<ClassicalKernel &>(root);

    assert(provided_captures.size() == kernel.capture_names.size()
           && provided_captures.size() == kernel.capture_types.size()
           && "capture length mismatch");

    // Synthesize constant captures
    for (size_t i = 0; i < provided_captures.size(); i++) {
        Bits *bits;
        if (!(bits = dynamic_cast<Bits *>(provided_captures[i].get()))) {
            // TODO: This either needs to be caught by the type system, or we
            //       need to allow you to do crazier stuff
            throw CompileException("Currently, " + ast_kind_name(AST_CLASSICAL)
                                   + " kernels can only capture bit[N]. Sorry!",
                                   std::move(root.dbg->copy()));
        }

        [[maybe_unused]] const BitType &bitType =
            dynamic_cast<const BitType &>(bits->getType());
        assert(bitType.dim->isConstant()
               && bitType.dim->offset == (DimVarValue)bits->bits.size()
               && "Type of Bits does not match its actual number of bits");

        ASTNode::Wires &dest = variable_wires[kernel.capture_names[i]];
        for (bool bit : bits->bits) {
            dest.push_back(net.get_constant(bit));
        }
    }

    #define CREATE_INPUTS(arg_ref, arg_idx) \
        BitType &arg_type = dynamic_cast<BitType &>(arg_ref); \
        assert(arg_type.dim->isConstant() \
               && "type variables remaining in compilation!"); \
        std::string &arg_name = kernel.arg_names[arg_idx]; \
        ASTNode::Wires &dest = variable_wires[arg_name]; \
        for (ssize_t j = 0; j < arg_type.dim->offset; j++) { \
            dest.push_back(net.create_pi("_" + arg_name + "_" \
                           + std::to_string(j))); \
        }

    FuncType &funcType = dynamic_cast<FuncType &>(*kernel.type);
    if (kernel.arg_names.empty()) {
        // No wires to add
    } else if (kernel.arg_names.size() == 1) {
        CREATE_INPUTS(*funcType.lhs, 0)
    } else {
        TupleType &args_type = dynamic_cast<TupleType &>(*funcType.lhs);
        for (size_t i = 0; i < kernel.arg_names.size(); i++) {
            CREATE_INPUTS(*args_type.types[i], i)
        }
    }
}

bool ClassicalNetlistVisitor::visitNonClassicalNode(ASTVisitContext &ctx, ASTNode &node) {
    throw CompileException("How did a non-" + ast_kind_name(AST_CLASSICAL)
                           + " node end up in a " + ast_kind_name(AST_CLASSICAL)
                           + " AST?",
                           std::move(node.dbg->copy()));
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, Variable &var) {
    if (!variable_wires.count(var.name)) {
        throw CompileException("Variable " + var.name + " does not have a "
                               "wire. Type checking bug?",
                               std::move(var.dbg->copy()));
    }
    var.getWires() = variable_wires[var.name];
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, BitUnaryOp &unOp) {
    ASTNode::Wires &operand = unOp.operand->getWires();
    unOp.getWires().clear();
    for (size_t i = 0; i < operand.size(); i++) {
        switch (unOp.op) {
        case BIT_NOT:
            unOp.getWires().push_back(net.create_not(operand[i]));
            break;
        default:
            throw CompileException("Unknown bit op " + std::to_string(unOp.op),
                                   std::move(unOp.dbg->copy()));
        }
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, BitBinaryOp &binOp) {
    ASTNode::Wires &lhs = binOp.left->getWires();
    ASTNode::Wires &rhs = binOp.right->getWires();
    if (bit_op_is_broadcast(binOp.op) && lhs.size() != rhs.size()) {
        throw CompileException("Bit size mismatch " + std::to_string(lhs.size())
                               + " != " + std::to_string(rhs.size()) + ". Type "
                               "checking bug?",
                               std::move(binOp.dbg->copy()));
    }

    if (bit_op_is_broadcast(binOp.op)) {
        binOp.getWires().clear();
        for (size_t i = 0; i < lhs.size(); i++) {
            switch (binOp.op) {
            case BIT_AND:
                binOp.getWires().push_back(net.create_and(lhs[i], rhs[i]));
                break;
            case BIT_OR:
                binOp.getWires().push_back(net.create_or(lhs[i], rhs[i]));
                break;
            case BIT_XOR:
                binOp.getWires().push_back(net.create_xor(lhs[i], rhs[i]));
                break;
            default:
                throw CompileException("Unknown bit op " + std::to_string(binOp.op),
                                       std::move(binOp.dbg->copy()));
            }
        }
    } else {
        switch (binOp.op) {
        case BIT_ROTL:
            binOp.getWires() = bit_rotate(net, lhs, rhs, true);
            break;
        case BIT_ROTR:
            binOp.getWires() = bit_rotate(net, lhs, rhs, false);
            break;
        default:
            throw CompileException("Unknown bit op " + std::to_string(binOp.op),
                                   std::move(binOp.dbg->copy()));
        }
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) {
    ASTNode::Wires &operand_wires = reduceOp.operand->getWires();

    reduceOp.getWires().clear();
    switch (reduceOp.op) {
    case BIT_AND:
        reduceOp.getWires().push_back(net.create_nary_and(operand_wires));
        break;
    case BIT_XOR:
        reduceOp.getWires().push_back(net.create_nary_xor(operand_wires));
        break;
    default:
        throw CompileException("Unknown bit op " + std::to_string(reduceOp.op),
                               std::move(reduceOp.dbg->copy()));
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, BitConcat &concat) {
    concat.getWires() = concat.left->getWires();
    for (auto wire : concat.right->getWires()) {
        concat.getWires().push_back(wire);
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, BitRepeat &repeat) {
    assert(repeat.amt->isConstant()
           && "Dimvars snuck into repeat lowering");

    repeat.getWires().clear();
    ASTNode::Wires &bits_wires = repeat.bits->getWires();
    for (DimVarValue i = 0; i < repeat.amt->offset; i++) {
        repeat.getWires().insert(repeat.getWires().end(),
                                 bits_wires.begin(),
                                 bits_wires.end());
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, ModMulOp &mul) {
    assert(mul.x->isConstant()
           && mul.j->isConstant()
           && mul.modN->isConstant()
           && "Dimvars snuck into modular multiplication lowering!");
    assert(mul.x->offset > 0
           && mul.j->offset >= 0
           && mul.modN->offset >= 2
           && "x, j, modN out of range");

    uint64_t x = (uint64_t)mul.x->offset;
    uint64_t j = (uint64_t)mul.j->offset;
    uint64_t N = (uint64_t)mul.modN->offset;

    // TODO: should use llvm::APInt here in case N is very large. For now, we
    //       are only simulating, so it's probably fine as-is
    // Go ahead and do the repeated squaring here, classically
    uint64_t x_2j_modN = x % N;
    for (uint64_t i = 1; i <= j; i++) {
        x_2j_modN = (x_2j_modN * x_2j_modN) % N;
    }

    ASTNode::Wires &y_wires = mul.y->getWires();
    size_t n_bits = y_wires.size();

    ASTNode::Wires x_wires;
    for (size_t i = 0; i < n_bits; i++) {
        // Use the same endianness as Tweedledum
        bool bit = (x_2j_modN >> i) & 0x1;
        x_wires.push_back(net.get_constant(bit));
    }

    std::vector<bool> N_as_bits;
    for (size_t i = 0; i < y_wires.size(); i++) {
        bool bit = (N >> i) & 0x1;
        N_as_bits.push_back(bit);
    }

    // Tweedledum uses the opposite endianness as we do, so reverse the input qubits
    ASTNode::Wires y_wires_rev(y_wires.rbegin(), y_wires.rend());
    // Overwrites x_wires with the result
    mockturtle::modular_multiplication_inplace(net, x_wires, y_wires_rev, N_as_bits);
    // Also reverse the output
    ASTNode::Wires result_wires(x_wires.rbegin(), x_wires.rend());
    mul.getWires() = result_wires;
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, BitLiteral &bitLit) {
    assert(bitLit.val->isConstant()
           && bitLit.n_bits->isConstant()
           && "Dimvars snuck into bit literal lowering!");

    // TODO: what about when the requested value is larger than the host
    //       wordsize? llvm::APInt may be helpful here
    DimVarValue n_bits = bitLit.n_bits->offset;
    DimVarValue val = bitLit.val->offset;
    bitLit.getWires().clear();
    for (DimVarValue i = 0; i < n_bits; i++) {
        bool bit = (val >> (n_bits-i-1)) & 1;
        bitLit.getWires().push_back(net.get_constant(bit));
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, Slice &access) {
    assert(access.lower->isConstant()
           && access.upper->isConstant()
           && "Dimvars snuck into bit access lowering!");

    ASTNode::Wires &val_wires = access.val->getWires();
    assert(access.lower->offset >= 0
           && access.upper->offset >= 0
           && access.lower->offset <= access.upper->offset
           && (size_t)access.upper->offset <= val_wires.size()
           && "Bit access out of bounds. Why did type checking not catch this?");

    ASTNode::Wires &access_wires = access.getWires();
    access_wires.clear();
    access_wires.insert(access_wires.begin(),
                        val_wires.begin() + access.lower->offset,
                        val_wires.begin() + access.upper->offset);
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, Assign &assign) {
    assign.value->visit(ctx, *this);
    variable_wires[assign.target] = assign.value->getWires();
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, DestructAssign &dassign) {
    dassign.value->visit(ctx, *this);

    assert(dassign.targets.size() == dassign.value->getWires().size()
           && "Mismatch in dimension of lhs and rhs for destructuring "
              "assignment");

    for (size_t i = 0; i < dassign.targets.size(); i++) {
        auto &this_var_wires = variable_wires[dassign.targets[i]];
        this_var_wires.clear();
        this_var_wires.push_back(dassign.value->getWires()[i]);
    }
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, Return &ret) {
    ret.getWires() = ret.value->getWires();
    return true;
}

bool ClassicalNetlistVisitor::visit(ASTVisitContext &ctx, Kernel &kernel) {
    ClassicalKernel &revkernel = dynamic_cast<ClassicalKernel &>(kernel);
    if (revkernel.body.empty()) {
        throw CompileException("Kernel with empty body",
                               std::move(revkernel.dbg->copy()));
    }
    Return &ret = dynamic_cast<Return &>(*revkernel.body[revkernel.body.size()-1]);
    ASTNode::Wires &kern_wires = revkernel.getWires();
    kern_wires = ret.getWires();

    // TODO: is this always a safe cast?
    BitType &outputType = dynamic_cast<BitType &>(*dynamic_cast<FuncType &>(*revkernel.type).rhs);
    assert(outputType.dim->isConstant() && "dimvars have snuck into return values?");
    if ((DimVarValue)kern_wires.size() != outputType.dim->offset) {
        throw CompileException("Number of kernel output wires "
                               + std::to_string(kern_wires.size()) + " does "
                               "not match output type " + outputType.toString(),
                               std::move(revkernel.dbg->copy()));
    }

    output_wires.clear();
    for (size_t i = 0; i < kern_wires.size(); i++) {
        output_wires.push_back(net.create_po(kern_wires[i],
                                             "ret_" + std::to_string(i)));
    }
    return true;
}
