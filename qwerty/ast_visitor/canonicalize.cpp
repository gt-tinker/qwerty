#include "ast.hpp"
#include "ast_visitor.hpp"

bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, Adjoint &adj) {
    // ~~f -> f
    if (Adjoint *child_adj = dynamic_cast<Adjoint *>(adj.operand.get())) {
        ctx.ptr = std::move(child_adj->operand);
        return false;
    // ~b.flip -> b.flip
    } else if (dynamic_cast<Flip *>(adj.operand.get())) {
        ctx.ptr = std::move(adj.operand);
        return false;
    // ~id -> id
    } else if (dynamic_cast<Identity *>(adj.operand.get())) {
        ctx.ptr = std::move(adj.operand);
        return false;
    // ~b.rotate(theta) -> b.rotate(-theta)
    } else if (Rotate *rot = dynamic_cast<Rotate *>(adj.operand.get())) {
        std::unique_ptr<FloatNeg> neg_angle =
            std::make_unique<FloatNeg>(adj.dbg->copy(),
                                       std::move(rot->theta));
        rot->theta = std::move(neg_angle);
        ctx.ptr = std::move(adj.operand);
        return false;
    // ~(b1 >> b2) -> b2 >> b1
    } else if (BasisTranslation *btrans =
            dynamic_cast<BasisTranslation *>(adj.operand.get())) {
        btrans->basis_in.swap(btrans->basis_out);
        ctx.ptr = std::move(adj.operand);
        return false;
    } else {
        return true;
    }
}

bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, Pred &pred) {
    if (pred.order == PRED_ORDER_UNKNOWN) {
        // Maybe we re-canonicalized this out of nowhere and type checking
        // hasn't run yet to determine its order
        return true;
    }

    // std[N] & f -> id[N] + f
    // TODO: this can be generalized further to remove fully-spanning bases
    //       entirely, even if they are tensored with other stuff
    const BasisType *basis_type;
    if (pred.basis->hasType()
            && (basis_type = dynamic_cast<const BasisType *>(&pred.basis->getType()))
            && basis_type->span.fullySpans()) {
        std::unique_ptr<ASTNode> id =
            std::make_unique<Identity>(
                pred.dbg->copy(),
                basis_type->dim->copy());
        ctx.ptr = std::make_unique<BiTensor>(pred.dbg->copy(),
            pred.order == PRED_ORDER_B_U? std::move(id) : std::move(pred.body),
            pred.order == PRED_ORDER_B_U? std::move(pred.body) : std::move(id));
        return false;
    }

    if (BasisTranslation *btrans =
            dynamic_cast<BasisTranslation *>(pred.body.get())) {
        std::unique_ptr<ASTNode> lhs =
            std::make_unique<BiTensor>(
                pred.dbg->copy(),
                pred.order == PRED_ORDER_B_U? pred.basis->copy()
                                          : std::move(btrans->basis_in),
                pred.order == PRED_ORDER_B_U? std::move(btrans->basis_in)
                                          : pred.basis->copy());
        std::unique_ptr<ASTNode> rhs =
            std::make_unique<BiTensor>(
                pred.dbg->copy(),
                pred.order == PRED_ORDER_B_U? std::move(pred.basis)
                                          : std::move(btrans->basis_out),
                pred.order == PRED_ORDER_B_U? std::move(btrans->basis_out)
                                          : std::move(pred.basis));
        ctx.ptr = std::make_unique<BasisTranslation>(std::move(btrans->dbg),
            std::move(lhs), std::move(rhs));
        return false;
    } else {
        return true;
    }
}

bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, FloatNeg &neg) {
    if (FloatLiteral *flit = dynamic_cast<FloatLiteral *>(neg.operand.get())) {
        flit->value = -flit->value;
        ctx.ptr = std::move(neg.operand);
        return false;
    } else if (FloatNeg *fneg = dynamic_cast<FloatNeg *>(neg.operand.get())) {
        ctx.ptr = std::move(fneg->operand);
        return false;
    } else {
        return true;
    }
}

namespace {
bool simulate_float_bin_op(FloatOp op, double left, double right, double &result) {
    switch (op) {
        case FLOAT_DIV:
            if (right) {
                result = left / right;
                return true;
            } else {
                // Undefined behavior... or not... not gonna touch this with a
                // 1000ft pole
                return false;
            }
        case FLOAT_POW:
            result = std::pow(left, right);
            if (std::isfinite(result)) {
                return true;
            } else {
                return false;
            }
        case FLOAT_MUL:
            result = left * right;
            return true;
        default: assert(0 && "Missing float bin op to simulate"); return 0;
    }
}
} // namespace

bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, FloatBinaryOp &bin) {
    FloatLiteral *lhs, *rhs;
    if ((lhs = dynamic_cast<FloatLiteral *>(bin.left.get()))
            && (rhs = dynamic_cast<FloatLiteral *>(bin.right.get()))) {
        double res;
        if (!simulate_float_bin_op(bin.op, lhs->value, rhs->value, res)) {
            // Not touching this one
            return true;
        }
        ctx.ptr = std::make_unique<FloatLiteral>(bin.dbg->copy(), res);
        return false;
    } else {
        return true;
    }
}

// This works in concert with the BroadcastTensor visitor in
// EvalDimVarExprVisitor
bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, BroadcastTensor &broad_tensor) {
    ASTNode *value = broad_tensor.value.get();

    QubitLiteral *vector;
    BuiltinBasis *std_basis;
    Identity *id;
    Discard *discard;
    Measure *meas;
    if ((vector = dynamic_cast<QubitLiteral *>(value))) {
        *vector->type.dim *= *broad_tensor.factor;
        ctx.ptr = std::move(broad_tensor.value);
        return false;
    } else if ((std_basis = dynamic_cast<BuiltinBasis *>(value))) {
        *std_basis->type.dim *= *broad_tensor.factor;
        ctx.ptr = std::move(broad_tensor.value);
        return false;
    } else if ((id = dynamic_cast<Identity *>(value))) {
        *static_cast<QubitType *>(id->type.lhs.get())->dim *= *broad_tensor.factor;
        *static_cast<QubitType *>(id->type.rhs.get())->dim *= *broad_tensor.factor;
        ctx.ptr = std::move(broad_tensor.value);
        return false;
    } else if ((discard = dynamic_cast<Discard *>(value))) {
        *static_cast<QubitType *>(discard->type.lhs.get())->dim *= *broad_tensor.factor;
        ctx.ptr = std::move(broad_tensor.value);
        return false;
    } else if ((meas = dynamic_cast<Measure *>(value))
               && (std_basis = dynamic_cast<BuiltinBasis *>(meas->basis.get()))) {
        *std_basis->type.dim *= *broad_tensor.factor;
        ctx.ptr = std::move(broad_tensor.value);
        return false;
    }

    return true;
}

bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, LiftBits &lift) {
    // TODO: turn lifted bit literal into a qubit literal

    const BitType *bit_type = dynamic_cast<const BitType *>(&lift.bits->getType());
    assert(bit_type && "operand to liftbits is not a bit[N]. "
                       "shouldn't typechecking catch this?");
    const DimVarExpr &dim = *bit_type->dim;
    std::unique_ptr<ASTNode> replacement = std::make_unique<Pipe>(
        std::move(lift.dbg->copy()),
        std::make_unique<QubitLiteral>(std::move(lift.dbg->copy()),
                                       PLUS,
                                       Z,
                                       std::move(dim.copy())),
        std::make_unique<Prepare>(std::move(lift.dbg->copy()),
                                  std::move(lift.bits)));

    ctx.ptr = std::move(replacement);
    return false;
}

bool CanonicalizeVisitor::visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) {
    assert(fdve.value->isConstant() && "dimvar snuck into FloatDimVarExpr canonicalization");
    double float_const = (double)fdve.value->offset;
    ctx.ptr = std::make_unique<FloatLiteral>(std::move(fdve.dbg->copy()), float_const);
    return false;
}
