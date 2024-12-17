#include "ast.hpp"
#include "ast_visitor.hpp"

#define RETHROW_AS_TYPE_EXCEPTION(offender, code) \
    try { \
        DimVarValues dvvs = dimvar_values; \
        DimVarValues &offender_dvvs = offender.getScopedDimvars(); \
        dvvs.insert(offender_dvvs.begin(), offender_dvvs.end()); \
        code; \
    } catch (MissingDimVarException &err) { \
        throw TypeException("Type variable " + err.missingDimvar + " undefined", \
                            std::move(offender.dbg->copy())); \
    } catch (NegativeDimVarExprException &err) { \
        if (err.offendingDimvars.empty()) { \
            throw TypeException("Constant expression is negative", \
                                std::move(offender.dbg->copy())); \
        } else { \
            std::string offenders; \
            for (DimVar dimvar : err.offendingDimvars) { \
                if (!offenders.empty()) { \
                    offenders += ", "; \
                } \
                offenders += dimvar; \
            } \
            throw TypeException("Constant expression is negative due to " \
                                "choice of type variables " \
                                + offenders, \
                                std::move(offender.dbg->copy())); \
        } \
    }

bool EvalDimVarExprVisitor::visitNode(ASTVisitContext &ctx, ASTNode &node) {
    node.inheritScopedDimvars(ctx.parent);
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Kernel &kernel) {
    if (!visitNode(ctx, kernel)) {
        return false;
    }

    RETHROW_AS_TYPE_EXCEPTION(kernel, kernel.type->evalDimVarExprs(dvvs, false));
    if (kernel.type->isFurled()) {
        kernel.type = std::move(kernel.type->unfurl());
    }

    for (size_t i = 0; i < kernel.capture_types.size(); i++) {
        RETHROW_AS_TYPE_EXCEPTION(kernel, kernel.capture_types[i]->evalDimVarExprs(dvvs, false));
        if (kernel.capture_types[i]->isFurled()) {
            kernel.capture_types[i] = std::move(kernel.capture_types[i]->unfurl());
        }
    }

    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Pipe &pipe) {
    if (!visitNode(ctx, pipe)) {
        return false;
    }

    // We need to treat repeat nodes specially: pipe repeat->body into itself ub times
    if (Repeat *repeat = dynamic_cast<Repeat *>(pipe.right.get())) {
        RETHROW_AS_TYPE_EXCEPTION((*repeat), repeat->ub->eval(dvvs, false));

        assert(repeat->ub->isConstant() && "eval() left a dimvarexpr non-constant?? how?");
        assert(repeat->ub->offset >= 0 && "eval() left a dimvarexpr negative?? how?");

        if (!repeat->ub->offset) {
            // Special case: don't apply repeat->body at all
            ctx.ptr = std::move(pipe.left);
            return false;
        } else {
            std::unique_ptr<ASTNode> body = std::move(repeat->body->copy());
            body->setScopedDimvar(repeat->loopvar, 0);
            std::unique_ptr<ASTNode> root = std::make_unique<Pipe>(std::move(pipe.dbg->copy()),
                                                                   std::move(pipe.left),
                                                                   std::move(body));
            for (DimVarValue i = 1; i < repeat->ub->offset; i++) {
                std::unique_ptr<ASTNode> body = std::move(repeat->body->copy());
                body->setScopedDimvar(repeat->loopvar, i);
                root = std::move(std::make_unique<Pipe>(std::move(repeat->dbg->copy()),
                                                        std::move(root),
                                                        std::move(body)));
            }
            ctx.ptr = std::move(root);
            return false;
        }
    } else {
        // No dimvarexprs to expand if the right hand side isn't a Repeat
        return true;
    }
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Instantiate &inst) {
    if (!visitNode(ctx, inst)) {
        return false;
    }
    for (size_t i = 0; i < inst.instance_vals.size(); i++) {
        RETHROW_AS_TYPE_EXCEPTION(inst, inst.instance_vals[i]->eval(dvvs, false))
    }
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Repeat &repeat) {
    if (!visitNode(ctx, repeat)) {
        return false;
    }
    throw TypeException("A repeat construct must follow a pipe",
                        std::move(repeat.dbg->copy()));
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, BroadcastTensor &broad_tensor) {
    if (!visitNode(ctx, broad_tensor)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(broad_tensor, broad_tensor.factor->eval(dvvs, false))

    // something[0] needs to be replaced with ()
    Measure *meas;
    if (!broad_tensor.factor->offset) {
        ctx.ptr = std::make_unique<TupleLiteral>(std::move(broad_tensor.dbg->copy()));
        return false;
    } else if (broad_tensor.factor->offset == 1) {
        ctx.ptr = std::move(broad_tensor.value);
        return false;
    } else if (dynamic_cast<QubitLiteral *>(broad_tensor.value.get())
               || dynamic_cast<BuiltinBasis *>(broad_tensor.value.get())
               || dynamic_cast<Identity *>(broad_tensor.value.get())
               || dynamic_cast<Discard *>(broad_tensor.value.get())
               || ((meas = dynamic_cast<Measure *>(broad_tensor.value.get()))
                   && dynamic_cast<BuiltinBasis *>(meas->basis.get()))) {
        // Canonicalization will take care of these guys more efficiently
        return true;
    } else { // factor > 1
        // Expand to chain of BiTensors
        std::unique_ptr<ASTNode> root = std::make_unique<BiTensor>(
            std::move(broad_tensor.dbg->copy()),
            std::move(broad_tensor.value->copy()),
            std::move(broad_tensor.value->copy()));
        for (DimVarValue i = 3; i <= broad_tensor.factor->offset; i++) {
            root = std::make_unique<BiTensor>(
                std::move(broad_tensor.dbg->copy()),
                std::move(root),
                std::move(broad_tensor.value->copy()));
        }
        ctx.ptr = std::move(root);
        return false;
    }
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, QubitLiteral &lit) {
    if (!visitNode(ctx, lit)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(lit, lit.type.evalDimVarExprs(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, BuiltinBasis &std) {
    if (!visitNode(ctx, std)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(std, std.type.evalDimVarExprs(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Identity &id) {
    if (!visitNode(ctx, id)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(id, id.type.evalDimVarExprs(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Discard &discard) {
    if (!visitNode(ctx, discard)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(discard, discard.type.evalDimVarExprs(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, BitRepeat &repeat) {
    if (!visitNode(ctx, repeat)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(repeat, repeat.amt->eval(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, ModMulOp &mul) {
    if (!visitNode(ctx, mul)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(mul, mul.x->eval(dvvs, false))
    RETHROW_AS_TYPE_EXCEPTION(mul, mul.j->eval(dvvs, false))
    RETHROW_AS_TYPE_EXCEPTION(mul, mul.modN->eval(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, BitLiteral &bitLit) {
    if (!visitNode(ctx, bitLit)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(bitLit, bitLit.val->eval(dvvs, false))
    RETHROW_AS_TYPE_EXCEPTION(bitLit, bitLit.n_bits->eval(dvvs, false))
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, Slice &slice) {
    if (!visitNode(ctx, slice)) {
        return false;
    }
    if (slice.lower) {
        RETHROW_AS_TYPE_EXCEPTION(slice, slice.lower->eval(dvvs, false))
    }
    if (slice.upper) {
        RETHROW_AS_TYPE_EXCEPTION(slice, slice.upper->eval(dvvs, false))
    }
    return true;
}

bool EvalDimVarExprVisitor::visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) {
    if (!visitNode(ctx, fdve)) {
        return false;
    }
    RETHROW_AS_TYPE_EXCEPTION(fdve, fdve.value->eval(dvvs, false))
    return true;
}

#undef RETHROW_AS_TYPE_EXCEPTION
