#include "ast.hpp"
#include "ast_visitor.hpp"

void BaseTypeCheckVisitor::validateSlice(Slice &slice,
                                         DimVarValue dim,
                                         DimVarValue &n_elem,
                                         DimVarValue &start_idx) {

    if (!slice.lower) {
        slice.lower = std::make_unique<DimVarExpr>("", 0);
    }
    assert(slice.lower->isConstant() && "dimvars snuck into slice lower");
    if (!slice.upper) {
        slice.upper = std::make_unique<DimVarExpr>("", dim);
    }
    assert(slice.upper->isConstant() && "dimvars snuck into slice upper");

    if (slice.lower->offset < 0) {
        throw TypeException("Slice index negative: "
                            + std::to_string(slice.lower->offset) + " < 0",
                            std::move(slice.dbg->copy()));
    }
    if (slice.upper->offset < 0) {
        throw TypeException("Slice index negative: "
                            + std::to_string(slice.upper->offset) + " < 0",
                            std::move(slice.dbg->copy()));
    }

    if (slice.lower->offset >= slice.upper->offset) {
        throw TypeException("Slice lower bound larger than upper bound: "
                            + std::to_string(slice.lower->offset)
                            + " >= "
                            + std::to_string(slice.upper->offset),
                            std::move(slice.dbg->copy()));
    }

    if (slice.upper->offset > dim) {
        throw TypeException("Slice index out of bounds: "
                            + std::to_string(slice.upper->offset)
                            + " > " + std::to_string(dim),
                            std::move(slice.dbg->copy()));
    }

    n_elem = slice.upper->offset - slice.lower->offset;
    start_idx = slice.lower->offset;
}

bool BaseTypeCheckVisitor::visit(ASTVisitContext &ctx, Variable &var) {
    if (!variables.count(var.name)) {
        throw TypeException("Variable " + var.name + " not defined",
                            std::move(var.dbg->copy()));
    }

    var.type = std::move(variables.at(var.name)->copy());

    // Insert variable name into vars_used once visited
    if (var.type->isLinear() && !linear_vars_used.insert(var.name).second) {
        throw TypeException("Variable " + var.name + " already in use!",
                            std::move(var.dbg->copy()));
    }

    return true;
}

bool BaseTypeCheckVisitor::visit(ASTVisitContext &ctx, Assign &assign) {
    if (variables.count(assign.target)) {
        throw TypeException("All variables are immutable and cannot be "
                            "shadowed: Variable " + assign.target + " was "
                            "already defined.", std::move(assign.dbg->copy()));
    }

    assign.value->visit(ctx, *this);
    assign.type = std::move(assign.value->getType().copy());
    variables.emplace(assign.target, assign.type->copy());
    return true;
}

bool BaseTypeCheckVisitor::visit(ASTVisitContext &ctx, DestructAssign &dassign) {
    std::unordered_set<std::string> unique_targets(dassign.targets.begin(),
                                                   dassign.targets.end());
    if (unique_targets.size() != dassign.targets.size()) {
        throw TypeException("All names on the left-hand side of an assignment "
                            "must be distinct.",
                            std::move(dassign.dbg->copy()));
    }

    for (std::string &tgt : dassign.targets) {
        if (variables.count(tgt)) {
            throw TypeException("All variables are immutable and cannot be "
                                "shadowed: Variable " + tgt + " was "
                                "already defined.",
                                std::move(dassign.dbg->copy()));
        }
    }

    dassign.value->visit(ctx, *this);
    const Type &ty = dassign.value->getType();

    std::vector<std::unique_ptr<Type>> lhs_types;
    if (const TupleType *tuple_ty = dynamic_cast<const TupleType *>(&ty)) {
        for (size_t i = 0; i < tuple_ty->types.size(); i++) {
            lhs_types.push_back(tuple_ty->types[i]->copy());
        }
    } else if (const QubitType *qubit_ty =
                   dynamic_cast<const QubitType *>(&ty)) {
        assert(qubit_ty->dim->isConstant()
               && "dimvars snuck into destruct type checking");
        for (DimVarValue i = 0; i < qubit_ty->dim->offset; i++) {
            lhs_types.push_back(std::make_unique<QubitType>(
                std::make_unique<DimVarExpr>("", 1)));
        }
    } else if (const BitType *bit_ty =
                   dynamic_cast<const BitType *>(&ty)) {
        assert(bit_ty->dim->isConstant()
               && "dimvars snuck into destruct type checking");
        for (DimVarValue i = 0; i < bit_ty->dim->offset; i++) {
            lhs_types.push_back(std::make_unique<BitType>(
                std::make_unique<DimVarExpr>("", 1)));
        }
    } else {
        throw TypeException("Destructuring assignment is not supported for "
                            "type " + ty.toString(),
                            std::move(dassign.dbg->copy()));
    }

    size_t lhs_dim = dassign.targets.size();
    size_t rhs_dim = lhs_types.size();
    if (lhs_dim != rhs_dim) {
        throw TypeException("Wrong number of names on the left-hand side. "
                            "Expected " + std::to_string(rhs_dim)
                            + " because right-hand side has type "
                            + ty.toString() + " but instead got "
                            + std::to_string(lhs_dim)
                            + " names on left-hand side.",
                            std::move(dassign.dbg->copy()));
    }

    dassign.type = std::move(ty.copy());
    for (size_t i = 0; i < lhs_dim; i++) {
        std::string &tgt = dassign.targets[i];
        [[maybe_unused]] auto pair =
            variables.emplace(tgt, std::move(lhs_types[i]));
        assert(pair.second && "Duplicate variable names?");
    }
    return true;
}

bool BaseTypeCheckVisitor::visit(ASTVisitContext &ctx, Return &ret) {
    ret.value->visit(ctx, *this);
    ret.type = std::move(ret.value->getType().copy());
    return true;
}

void BaseTypeCheckVisitor::init(ASTNode &root) {
    // Need to do a tiny bit of preorder traversal just for the kernel node to
    // set up for type checking my body
    Kernel *kernel = dynamic_cast<Kernel *>(&root);

    if (kernel) {
        // TODO: do we even need these? they don't seem to be used
        dimvars.insert(kernel->dimvars.begin(), kernel->dimvars.end());

        FuncType &funcType = dynamic_cast<FuncType &>(*kernel->type);
        if (kernel->arg_names.size() == 0) {
            // No arguments to add to typing context
        } else if (kernel->arg_names.size() == 1) {
            if (!variables.emplace(kernel->arg_names[0], funcType.lhs->copy()).second) {
                throw TypeException("Argument " + kernel->arg_names[0] + " already defined",
                                    std::move(kernel->dbg->copy()));
            }
        } else {
            TupleType &tuple = dynamic_cast<TupleType &>(*funcType.lhs);
            for (size_t i = 0; i < tuple.types.size(); i++) {
                if (!variables.emplace(kernel->arg_names[i], tuple.types[i]->copy()).second) {
                    throw TypeException("Argument " + kernel->arg_names[i] + " already defined",
                                        std::move(kernel->dbg->copy()));
                }
            }
        }

        assert(kernel->capture_names.size() == kernel->capture_types.size()
               && "Number of capture names and types do not match");
        for (size_t i = 0; i < kernel->capture_names.size(); i++) {
            if (!variables.emplace(kernel->capture_names[i], kernel->capture_types[i]->copy()).second) {
                throw TypeException("Capture " + kernel->capture_names[i] + " already defined",
                                    std::move(kernel->dbg->copy()));
            }
        }
    } else {
        // Unit tests may call the type checker on raw expressions, so silently
        // proceed if the root node is not a Kernel
    }
}

bool BaseTypeCheckVisitor::visit(ASTVisitContext &ctx, Kernel &kernel) {
    FuncType &funcType = dynamic_cast<FuncType &>(*kernel.type);
    for (size_t i = 0; i < kernel.body.size(); i++) {
        ASTNode *node = kernel.body[i].get();
        node->visit(ctx, *this);

        Return *ret;
        if ((ret = dynamic_cast<Return *>(node))) {
            if (i+1 != kernel.body.size()) {
                throw TypeException("Return statement must be last statement of kernel",
                                    std::move(kernel.dbg->copy()));
            }
            // Similar to the S-Arrow rule in Chapter 15 of TAPL
            if (!(ret->getType() <= *funcType.rhs)) {
                throw TypeException("Type error for last statement. Function returns " +
                                    funcType.rhs->toString() + " but attempting to "
                                    "return " + ret->getType().toString(),
                                    std::move(ret->dbg->copy()));
            }
        } else if (i+1 == kernel.body.size()) {
            throw TypeException("Last statement of a kernel must be a return!",
                                std::move(kernel.dbg->copy()));
        }
    }

    // Linear typing verification / Make sure qubit types are used only once
    for (const std::pair<const std::string, std::unique_ptr<Type>> &var : variables) {
        if (var.second->isLinear()) {
            if (linear_vars_used.count(var.first) == 0) {
                throw TypeException("Variable " + var.first + " of type "
                                    + var.second->toString() + " includes a "
                                    "qubit, so it must be used exactly once!",
                                    std::move(kernel.dbg->copy()));
            }
        }
    }

    // TODO: Make this error message more helpful, e.g., tell you which line
    //       number the irreversible construct is on
    if (funcType.is_rev && !is_rev) {
        throw TypeException("Function " + kernel.name + "() calls "
                            "irreversible functions or uses irreversible "
                            "constructs. It cannot be defined as @reversible",
                            std::move(kernel.dbg->copy()));
    }

    return true;
}

///////////////////////// @QPU TYPE CHECKING /////////////////////////

bool QpuTypeCheckVisitor::visitNonQpuNode(ASTVisitContext &ctx, ASTNode &node) {
    throw TypeException("How is there a non-@qpu AST node "
                        + node.label() + " in a @qpu AST?",
                        std::move(node.dbg->copy()));
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Slice &slice) {
    slice.val->visit(ctx, *this);
    if (!slice.val->getType().isClassical()) {
        throw TypeException("Currently, you can slice only classical types",
                            std::move(slice.dbg->copy()));
    }

    const TupleType *tuple = dynamic_cast<const TupleType *>(&slice.val->getType());
    DimVarValue dim = tuple? tuple->types.size() : 1;
    DimVarValue n_elem, start_idx;
    validateSlice(slice, dim, n_elem, start_idx);
    if (!tuple) {
        slice.type = std::move(slice.val->getType().copy());
    } else {
        slice.type = std::move(tuple->slice(start_idx, n_elem));
    }
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Adjoint &adj) {
    adj.operand->visit(ctx, *this);
    if (!isQfunc(&adj.operand->getType(), *adj.dbg, true, nullptr)) {
        throw TypeException("Operand f of adjoint ~f must be rev_qfunc[N] (a "
                            "reversible function from qubit[N] to qubit[N]), "
                            "not " + adj.operand->getType().toString(),
                            std::move(adj.dbg->copy()));
    }
    adj.type = std::move(adj.operand->getType().copy());
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Prepare &prep) {
    prep.operand->visit(ctx, *this);
    std::unique_ptr<ASTNode> basis_operand;
    const BitType *bit_type = dynamic_cast<const BitType *>(&prep.operand->getType());
    const DimVarExpr *dim;
    if (bit_type) {
        dim = bit_type->dim.get();
    } else {
        basis_operand = std::move(wrapBasis(ctx, std::move(prep.operand),
                                            /*allow_empty=*/false));

        if (!singletonBasis(basis_operand.get())) {
            throw TypeException("A basis passed to .prep must be a singleton "
                                "basis",
                                std::move(prep.dbg->copy()));
        }

        const BasisType *basis_type = dynamic_cast<const BasisType *>(&basis_operand->getType());
        assert(basis_type && "wrapBasis() didn't return a basis, how?");
        dim = basis_type->dim.get();
    }

    if (basis_operand) {
        prep.operand = std::move(basis_operand);
    }
    prep.type = std::make_unique<FuncType>(
        std::make_unique<QubitType>(std::move(dim->copy())),
        std::make_unique<QubitType>(std::move(dim->copy())),
        /*is_rev=*/true);
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, LiftBits &lift) {
    lift.bits->visit(ctx, *this);
    const BitType *bit_type;
    if (!(bit_type = dynamic_cast<const BitType *>(&lift.bits->getType()))) {
        throw TypeException("The operand x of a bit lift x.q must have type "
                            "bit[N], not " + lift.bits->getType().toString(),
                            std::move(lift.dbg->copy()));
    }
    lift.type = std::make_unique<QubitType>(std::move(bit_type->dim->copy()));
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, EmbedClassical &embed) {
    if (!variables.count(embed.name)) {
        throw TypeException("Classical function named " + embed.name + " not defined",
                            std::move(embed.dbg->copy()));
    }

    if (embedding_kind_has_operand(embed.kind)) {
        if (embed.operand_name.empty()) {
            throw TypeException("Missing required operand for ."
                                + embedding_kind_name(embed.kind),
                                std::move(embed.dbg->copy()));
        }
        if (!variables.count(embed.operand_name)) {
            throw TypeException("Undefined classical function named "
                                + embed.operand_name
                                + " passed as operand",
                                std::move(embed.dbg->copy()));
        }
        if (embed.kind == EMBED_INPLACE
                && *variables.at(embed.name) != *variables.at(embed.operand_name)) {
            throw TypeException("Type of reverse function "
                                + embed.operand_name
                                + " does not match type of forward function "
                                + embed.name,
                                std::move(embed.dbg->copy()));
        }
    } else if (!embed.operand_name.empty()) {
        throw TypeException("Unexpected operand provided for ."
                            + embedding_kind_name(embed.kind),
                            std::move(embed.dbg->copy()));
    }

    Type &type = *variables.at(embed.name);
    embed.type = type.asEmbedded(*embed.dbg, embed.kind);
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Pipe &pipe) {
    pipe.left->visit(ctx, *this);
    pipe.right->visit(ctx, *this);
    const FuncType *funcType;
    if (!(funcType = dynamic_cast<const FuncType *>(&pipe.right->getType()))) {
        throw TypeException("Right-hand side of pipe must be a function but got " +
                            pipe.right->getType().toString(),
                            std::move(pipe.dbg->copy()));
    }
    if (!(pipe.left->getType() <= *funcType->lhs)) {
        throw TypeException("Input to pipe " + pipe.left->getType().toString()
                            + " does not match right-hand input type "
                            + funcType->lhs->toString(),
                            std::move(pipe.dbg->copy()));
    }
    // For this function to be reversible, all functions it calls should be
    // reversible
    is_rev = is_rev && funcType->is_rev;
    pipe.type = std::move(funcType->rhs->copy());
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Instantiate &inst) {
    inst.var->visit(ctx, *this);
    if (!dynamic_cast<Variable *>(inst.var.get()) &&
            !dynamic_cast<EmbedClassical *>(inst.var.get())) {
        throw TypeException("Left-hand side of a instantiation must be a variable or "
                            "embedded classical function",
                            std::move(inst.dbg->copy()));
    }
    inst.type = std::move(inst.var->getType().copy());
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Pred &pred) {
    pred.basis->visit(ctx, *this);
    pred.body->visit(ctx, *this);
    DimVarExpr *qubit_dim;
    const Type &left_type = pred.basis->getType();
    const Type &right_type = pred.body->getType();
    if (pred.order == PRED_ORDER_UNKNOWN) {
        // No one has sorted out the order of operands yet
        if (isQfunc(&left_type, *pred.dbg, true, &qubit_dim)) {
            pred.order = PRED_ORDER_U_B;
            // Currently, there's a qfunc in the basis slot and (hopefully) a basis
            // in the body slot. Swap 'em
            pred.basis.swap(pred.body);
        } else if (isQfunc(&right_type, *pred.dbg, true, &qubit_dim)) {
            pred.order = PRED_ORDER_B_U;
        } else {
            throw TypeException("At least one operand of & must be a rev_qfunc. But "
                                "instead got " + left_type.toString() + " and "
                                + right_type.toString() + ", respectively",
                                std::move(pred.dbg->copy()));
        }
    } else if (!isQfunc(&right_type, *pred.dbg, true, &qubit_dim)) {
        // pred.order already set. Does it make sense?
        throw TypeException("The body operand must be a rev_qfunc. But"
                            "instead got " + right_type.toString(),
                            std::move(pred.dbg->copy()));
    }
    assert(qubit_dim->isConstant() && "dimvars snuck into pred lowering");

    std::unique_ptr<ASTNode> basis = std::move(wrapBasis(ctx, std::move(pred.basis),
                                                         /*allow_empty=*/false));
    const BasisType &basis_type = dynamic_cast<const BasisType &>(basis->getType());
    const DimVarExpr &basis_dim = *basis_type.dim;
    assert(basis_dim.isConstant() && "dimvars snuck into pred lowering");

    std::unique_ptr<DimVarExpr> new_dim = qubit_dim->copy();
    *new_dim += basis_dim;
    pred.type = std::make_unique<FuncType>(std::move(std::make_unique<QubitType>(new_dim->copy())),
                                          std::move(std::make_unique<QubitType>(new_dim->copy())),
                                          /*is_rev=*/true);
    pred.basis = std::move(basis);
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Conditional &cond) {
    cond.if_expr->visit(ctx, *this);

    // Check if if_expr is of type "bit"
    const BitType *bit_type;
    if (!(bit_type = dynamic_cast<const BitType *>(&cond.if_expr->getType()))) {
        throw TypeException("Operand of conditional test must be bit, not "
                            + cond.if_expr->getType().toString(),
                            std::move(cond.dbg->copy()));
    }

    std::unordered_set<std::string> backup = linear_vars_used;
    cond.then_expr->visit(ctx, *this);
    std::unordered_set<std::string> then_used = linear_vars_used;

    linear_vars_used = backup;
    cond.else_expr->visit(ctx, *this);

    if (then_used != linear_vars_used) {
        throw TypeException("If expression and Else expression do not use same variables but use qubits, \
                             this check says the qubits are not used at exactly once in both branches.",
                            std::move(cond.dbg->copy()));
    }

    // Check if types on "if" and "else" branches match
    if (cond.then_expr->getType() <= cond.else_expr->getType()) {
        cond.type = std::move(cond.else_expr->getType().copy());
    } else if (cond.else_expr->getType() <= cond.then_expr->getType()) {
        cond.type = std::move(cond.then_expr->getType().copy());
    } else {
        // TODO: try to find a common parent type? (Not useful until we have
        //       more complex subtyping)
        throw TypeException("'if' side of conditional is of type " +
                cond.then_expr->getType().toString() +
                " but 'else' side is of type " +
                cond.else_expr->getType().toString(),
                std::move(cond.dbg->copy()));
    }

    // A function with a conditional is not reversible
    is_rev = false;
    cond.type = std::move(cond.then_expr->getType().copy());
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Repeat &repeat) {
    // EvalDimVarExprVisitor should unroll these! What's going on here?
    throw TypeException("Unexpected Repeat construct. This is a compiler bug.",
                        std::move(repeat.dbg->copy()));
}

std::unique_ptr<Type> QpuTypeCheckVisitor::biTensorTypes(const Type *left,
                                                         const Type *right) {
    #define BI_TENSOR_PSEUDO_ARR_TYPE(name, extra_merge_code) \
        const name##Type *left##name##Type, *right##name##Type; \
        if ((left##name##Type = dynamic_cast<const name##Type *>(left)) \
                && (right##name##Type = dynamic_cast<const name##Type *>(right))) { \
            std::unique_ptr<DimVarExpr> new_dim = std::move(left##name##Type->dim->copy()); \
            *new_dim += *right##name##Type->dim; \
            std::unique_ptr<name##Type> merged = std::make_unique<name##Type>(std::move(new_dim)); \
            extra_merge_code \
            return merged; \
        } else if ((left##name##Type = dynamic_cast<const name##Type *>(left)) \
                && (tupleType = dynamic_cast<const TupleType *>(right)) \
                && tupleType->isUnit()) { \
            return left##name##Type->copy(); \
        } else if ((tupleType = dynamic_cast<const TupleType *>(left)) \
                && tupleType->isUnit() \
                && (right##name##Type = dynamic_cast<const name##Type *>(right))) { \
            return right##name##Type->copy(); \
        }

    const TupleType *tupleType;
    BI_TENSOR_PSEUDO_ARR_TYPE(Qubit, /* no extra merge code */)
    BI_TENSOR_PSEUDO_ARR_TYPE(Bit, /* no extra merge code */)
    BI_TENSOR_PSEUDO_ARR_TYPE(Basis,
        merged->span.clear();
        merged->span.append(leftBasisType->span);
        merged->span.append(rightBasisType->span);
    )

    // Else, we failed
    return nullptr;
}

bool QpuTypeCheckVisitor::onlyLiterals(ASTNode *node) {
    BiTensor *bi_tensor;
    BroadcastTensor *broad_tensor;
    TupleLiteral *tup;
    Phase *phase;
    return dynamic_cast<QubitLiteral *>(node)
           || dynamic_cast<BasisLiteral *>(node)
           || ((phase = dynamic_cast<Phase *>(node))
               && phase->only_literals)
           || ((tup = dynamic_cast<TupleLiteral *>(node))
               && tup->isUnit())
           || ((bi_tensor = dynamic_cast<BiTensor *>(node))
               && bi_tensor->only_literals)
           || ((broad_tensor = dynamic_cast<BroadcastTensor *>(node))
               && broad_tensor->only_literals);
}

bool QpuTypeCheckVisitor::singletonBasis(ASTNode *node) {
    BasisLiteral *basis_list;
    BiTensor *bi_tensor;
    BroadcastTensor *broad_tensor;
    return dynamic_cast<QubitLiteral *>(node)
           || ((basis_list = dynamic_cast<BasisLiteral *>(node))
               && basis_list->elts.size() == 1)
           || ((bi_tensor = dynamic_cast<BiTensor *>(node))
               && bi_tensor->singleton_basis)
           || ((broad_tensor = dynamic_cast<BroadcastTensor *>(node))
               && broad_tensor->singleton_basis);
}

PrimitiveBasis QpuTypeCheckVisitor::basisVectorPrimitiveBasisOrError(ASTNode *node, size_t vector_idx) {
    TupleLiteral *tup;
    if (QubitLiteral *lit = dynamic_cast<QubitLiteral *>(node)) {
        return lit->prim_basis;
    } else if (Phase *phase = dynamic_cast<Phase *>(node)) {
        return basisVectorPrimitiveBasisOrError(phase->value.get(), vector_idx);
    } else if (BiTensor *bi_tensor = dynamic_cast<BiTensor *>(node)) {
        PrimitiveBasis left_prim_basis = basisVectorPrimitiveBasisOrError(bi_tensor->left.get(), vector_idx);
        PrimitiveBasis right_prim_basis = basisVectorPrimitiveBasisOrError(bi_tensor->right.get(), vector_idx);

        if (left_prim_basis == (PrimitiveBasis)-1) {
            return right_prim_basis;
        } else if (right_prim_basis == (PrimitiveBasis)-1) {
            return left_prim_basis;
        } else if (left_prim_basis == right_prim_basis) {
            return left_prim_basis;
        } else {
            throw TypeException("Qubit literals in a basis literal must "
                                "not mix vectors of std[N], ij[N], "
                                "and pm[N], but a vector from "
                                + prim_basis_name(left_prim_basis) + "[N]"
                                " and a vector from "
                                + prim_basis_name(right_prim_basis) + "[N]"
                                " are mixed in the vector at index "
                                + std::to_string(vector_idx),
                                std::move(bi_tensor->dbg->copy()));
        }
    } else if (BroadcastTensor *broad_tensor = dynamic_cast<BroadcastTensor *>(node)) {
        return basisVectorPrimitiveBasisOrError(broad_tensor->value.get(), vector_idx);
    } else if ((tup = dynamic_cast<TupleLiteral *>(node)) && tup->isUnit()) {
        return (PrimitiveBasis)-1;
    } else {
        // TODO: Try and make this error message more helpful
        throw TypeException("Unknown syntax in basis vector at index "
                            + std::to_string(vector_idx) + " of basis literal",
                            std::move(node->dbg->copy()));
    }
}

llvm::APInt QpuTypeCheckVisitor::basisVectorEigenbitsOrError(ASTNode *node, size_t vector_idx) {
    TupleLiteral *tup;
    if (QubitLiteral *lit = dynamic_cast<QubitLiteral *>(node)) {
        assert(lit->type.dim->isConstant()
               && "Dimvars snuck into QubitLiteral in BasisLiteral type "
                  "checking");
        size_t n_bits = lit->type.dim->offset;
        llvm::APInt eigenbits(/*numBits=*/n_bits,
                              /*val=*/0,
                              /*isSigned=*/false);
        if (lit->eigenstate == MINUS) {
            eigenbits.setAllBits();
        }
        return eigenbits;
    } else if (Phase *phase = dynamic_cast<Phase *>(node)) {
        return basisVectorEigenbitsOrError(phase->value.get(), vector_idx);
    } else if (BiTensor *bi_tensor = dynamic_cast<BiTensor *>(node)) {
        llvm::APInt left_eigenbits = basisVectorEigenbitsOrError(bi_tensor->left.get(),
                                                                 vector_idx);
        llvm::APInt right_eigenbits = basisVectorEigenbitsOrError(bi_tensor->right.get(),
                                                                  vector_idx);
        return left_eigenbits.concat(right_eigenbits);
    } else if (BroadcastTensor *broad_tensor = dynamic_cast<BroadcastTensor *>(node)) {
        llvm::APInt value_eigenbits =
            basisVectorEigenbitsOrError(broad_tensor->value.get(), vector_idx);
        assert(broad_tensor->factor->isConstant()
               && "Dimvars snuck into BroadcastTensor in BasisLiteral type "
                  "checking");
        size_t n_bits = broad_tensor->factor->offset
                        * value_eigenbits.getBitWidth();
        llvm::APInt value_eigenbits_zext = value_eigenbits.zext(n_bits);
        llvm::APInt eigenbits(/*numBits=*/n_bits,
                              /*val=*/0,
                              /*isSigned=*/false);
        for (DimVarValue i = 0; i < broad_tensor->factor->offset; i++) {
            eigenbits |= value_eigenbits_zext
                         << value_eigenbits.getBitWidth()*i;
        }
        return eigenbits;
    } else if ((tup = dynamic_cast<TupleLiteral *>(node)) && tup->isUnit()) {
        llvm::APInt empty(/*numBits=*/0,
                          /*val=*/0,
                          /*isSigned=*/false);
        return empty;
    } else {
        // TODO: Try and make this error message more helpful
        throw TypeException("Unknown syntax in basis vector at index "
                            + std::to_string(vector_idx) + " of basis literal",
                            std::move(node->dbg->copy()));
    }
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, BiTensor &bi_tensor) {
    bi_tensor.left->visit(ctx, *this);
    bi_tensor.right->visit(ctx, *this);
    bi_tensor.only_literals = false;
    bi_tensor.singleton_basis = false;

    const Type *leftType = &bi_tensor.left->getType();
    const Type *rightType = &bi_tensor.right->getType();

    // Special case: can we wrapBasis one side?
    if (dynamic_cast<const BasisType *>(leftType)) {
        bi_tensor.right = std::move(wrapBasis(ctx, std::move(bi_tensor.right),
                                              /*allow_empty=*/true));
        rightType = &bi_tensor.right->getType();
    } else if (dynamic_cast<const BasisType *>(rightType)) {
        bi_tensor.left = std::move(wrapBasis(ctx, std::move(bi_tensor.left),
                                             /*allow_empty=*/true));
        leftType = &bi_tensor.left->getType();
    }

    // First, see if we can directly bi_tensor these types
    std::unique_ptr<Type> new_type = std::move(biTensorTypes(leftType, rightType));
    if (new_type) {
        bi_tensor.type = std::move(new_type);
        bi_tensor.only_literals = onlyLiterals(bi_tensor.left.get())
                                  && onlyLiterals(bi_tensor.right.get());
        bi_tensor.singleton_basis = singletonBasis(bi_tensor.left.get())
                                    && singletonBasis(bi_tensor.right.get());
        return true;
    }

    // Okay, now try bi_tensoring both sides of a function
    const FuncType *leftFuncType, *rightFuncType;
    if ((leftFuncType = dynamic_cast<const FuncType *>(leftType))
            && (rightFuncType = dynamic_cast<const FuncType *>(rightType))) {
        std::unique_ptr<Type> new_lhs, new_rhs;
        new_lhs = std::move(biTensorTypes(leftFuncType->lhs.get(), rightFuncType->lhs.get()));
        new_rhs = std::move(biTensorTypes(leftFuncType->rhs.get(), rightFuncType->rhs.get()));
        bool is_rev = leftFuncType->is_rev && rightFuncType->is_rev;

        if (new_lhs && new_rhs) {
            bi_tensor.type = std::make_unique<FuncType>(std::move(new_lhs),
                                                        std::move(new_rhs),
                                                        is_rev);
            return true;
        }
    }

    // Fallthrough is failure
    throw TypeException("I do not know how to tensor the types "
                        + leftType->toString() + " and " + rightType->toString(),
                        std::move(bi_tensor.dbg->copy()));
}

std::unique_ptr<Type> QpuTypeCheckVisitor::broadcastTensorType(const Type *type, const DimVarExpr &factor) {
    #define BROAD_TENSOR_PSEUDO_ARR_TYPE(name, extra_repeat_code) \
        const name##Type *as##name##Type; \
        if ((as##name##Type = dynamic_cast<const name##Type *>(type)) \
                && (factor.isConstant() || as##name##Type->dim->isConstant())) { \
            std::unique_ptr<DimVarExpr> new_dim = std::move(as##name##Type->dim->copy()); \
            *new_dim *= factor; \
            std::unique_ptr<name##Type> repeated = std::make_unique<name##Type>(std::move(new_dim)); \
            extra_repeat_code \
            return std::move(repeated); \
        }

    const TupleType *tupleType;
    if ((tupleType = dynamic_cast<const TupleType *>(type))
            && tupleType->isUnit()) {
        return std::move(tupleType->copy());
    }

    BROAD_TENSOR_PSEUDO_ARR_TYPE(Qubit, /* no extra repeat code */)
    BROAD_TENSOR_PSEUDO_ARR_TYPE(Bit, /* no extra repeat code */)
    BROAD_TENSOR_PSEUDO_ARR_TYPE(Basis,
        repeated->span.clear();
        for (DimVarValue i = 0; i < factor.offset; i++) {
            repeated->span.append(asBasisType->span);
        }
    )

    // Else, we failed
    return nullptr;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, BroadcastTensor &broad_tensor) {
    broad_tensor.value->visit(ctx, *this);
    broad_tensor.only_literals = false;
    broad_tensor.singleton_basis = false;
    const Type *valueType = &broad_tensor.value->getType();
    DimVarExpr &factor = *broad_tensor.factor;

    // First, see if we can directly broadcast tensor these types
    std::unique_ptr<Type> new_type = std::move(broadcastTensorType(valueType, factor));
    if (new_type) {
        broad_tensor.type = std::move(new_type);
        broad_tensor.only_literals = onlyLiterals(broad_tensor.value.get());
        broad_tensor.singleton_basis = singletonBasis(broad_tensor.value.get());
        return true;
    }

    // Okay, now try broadcast tensoring both sides of a function
    const FuncType *funcType;
    if ((funcType = dynamic_cast<const FuncType *>(valueType))) {
        std::unique_ptr<Type> new_lhs, new_rhs;
        new_lhs = std::move(broadcastTensorType(funcType->lhs.get(), factor));
        new_rhs = std::move(broadcastTensorType(funcType->rhs.get(), factor));

        if (new_lhs && new_rhs) {
            broad_tensor.type = std::move(std::make_unique<FuncType>(std::move(new_lhs),
                                                                     std::move(new_rhs),
                                                                     funcType->is_rev));
            return true;
        }
    }

    // Fallthrough is failure
    throw TypeException("I do not know how to tensor the type "
                        + valueType->toString() + " with itself "
                        + factor.toString() + " times. Perhaps the original "
                        + "dimension and the factor are both non-constant "
                        + "(not allowed)",
                        std::move(broad_tensor.dbg->copy()));
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Phase &phase) {
    phase.phase->visit(ctx, *this);
    phase.value->visit(ctx, *this);
    if (!dynamic_cast<const FloatType *>(&phase.phase->getType())) {
        throw TypeException("Expected Float as phase factor in phase "
                            "operator, not " +
                            phase.phase->getType().toString(),
                            std::move(phase.dbg->copy()));
    }

    const Type &value_type = phase.value->getType();
    if (dynamic_cast<const QubitType *>(&value_type)
            || isQfunc(&value_type, *phase.dbg, false, nullptr)) {
        phase.type = std::move(value_type.copy());
    } else {
        throw TypeException("Cannot apply a phase to " + value_type.toString(),
                            std::move(phase.dbg->copy()));
    }
    phase.only_literals = onlyLiterals(phase.value.get());
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, FloatNeg &neg) {
    neg.operand->visit(ctx, *this);
    if (!dynamic_cast<const FloatType *>(&neg.operand->getType())) {
        throw TypeException("Expected Float after `-' (negation), not " +
                            neg.operand->getType().toString(),
                            std::move(neg.dbg->copy()));
    }
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, FloatBinaryOp &bin) {
    bin.left->visit(ctx, *this);
    bin.right->visit(ctx, *this);

    if (!dynamic_cast<const FloatType *>(&bin.left->getType())) {
        throw TypeException("Expected Float as left operand of binary operation, not " +
                            bin.left->getType().toString(),
                            std::move(bin.dbg->copy()));
    }
    if (!dynamic_cast<const FloatType *>(&bin.right->getType())) {
        throw TypeException("Expected Float as right operand of binary operation, not " +
                            bin.right->getType().toString(),
                            std::move(bin.dbg->copy()));
    }
    return true;
}

// Nothing to check here
bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, QubitLiteral &lit) { return true; }
bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, FloatLiteral &float_) { return true; }
bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) { return true; }
bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Identity &id) { return true; }

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, BuiltinBasis &std) {
    assert(std.type.dim->isConstant() && "Dimvars snuck into basis type checking");
    std.type.span.clear();
    std.type.span.append(std::make_unique<FullSpan>(std.type.dim->offset));
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, TupleLiteral &tuple) {
    std::vector<std::unique_ptr<Type>> elt_types;
    for (size_t i = 0; i < tuple.elts.size(); i++) {
        tuple.elts[i]->visit(ctx, *this);
        std::unique_ptr<Type> canon_type = std::move(tuple.elts[i]->getType().canonicalize());
        const TupleType *sub_tuple;
        if ((sub_tuple = dynamic_cast<const TupleType *>(canon_type.get()))) {
            if (!sub_tuple->isUnit()) {
                throw TypeException("Tuples cannot be nested",
                                    std::move(tuple.dbg->copy()));
            }
            // Else (if Unit type), just don't put it in the list of types
        } else {
            elt_types.push_back(std::move(canon_type));
        }
    }
    tuple.type = std::move(std::make_unique<TupleType>(std::move(elt_types)));
    return true;
}

// Interpret individual basis states as basis lists
std::unique_ptr<ASTNode> QpuTypeCheckVisitor::wrapBasis(ASTVisitContext &ctx,
                                                        std::unique_ptr<ASTNode> node,
                                                        bool allow_empty) {
    const TupleType *tup;
    if (dynamic_cast<const BasisType *>(&node->getType())
            // Unit is an empty basis, effectively. However, many callers
            // expect something of BasisType returned
            || (allow_empty
                && (tup = dynamic_cast<const TupleType *>(&node->getType()))
                && tup->isUnit())) {
        return std::move(node);
    } else if (onlyLiterals(node.get())
               && dynamic_cast<const QubitType *>(&node->getType())) {
        std::unique_ptr<DebugInfo> dbg = std::move(node->dbg->copy());
        std::vector<std::unique_ptr<ASTNode>> elts;
        elts.push_back(std::move(node));
        std::unique_ptr<BasisLiteral> lit = std::move(std::make_unique<BasisLiteral>(
            std::move(dbg),
            std::move(elts)));
        visit(ctx, *lit);
        assert(dynamic_cast<const BasisType *>(&lit->getType())
               && "BasisLiteral is not a basis, how?");
        return std::move(lit);
    } else {
        throw TypeException("Expected basis but got " +
                            node->getType().toString(),
                            std::move(node->dbg->copy()));
    }
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, BasisTranslation &trans) {
    trans.basis_in->visit(ctx, *this);
    trans.basis_out->visit(ctx, *this);
    std::unique_ptr<ASTNode> basis_in = std::move(wrapBasis(ctx, std::move(trans.basis_in),
                                                            /*allow_empty=*/false));
    std::unique_ptr<ASTNode> basis_out = std::move(wrapBasis(ctx, std::move(trans.basis_out),
                                                             /*allow_empty=*/false));
    const BasisType &basis_in_type = dynamic_cast<const BasisType &>(basis_in->getType());
    const BasisType &basis_out_type = dynamic_cast<const BasisType &>(basis_out->getType());

    if (*basis_in_type.dim != *basis_out_type.dim) {
        throw TypeException("Input and output bases of basis translation "
                            "must have the same number of qubits, but "
                            + basis_in_type.dim->toString() + " != "
                            + basis_out_type.dim->toString(),
                            std::move(trans.dbg->copy()));
    }
    if (basis_in_type.span != basis_out_type.span) {
        throw TypeException("Input and output bases of basis translation "
                            "do not span the same subspace of qubits",
                            std::move(trans.dbg->copy()));
    }

    trans.type = std::move(std::make_unique<FuncType>(std::move(std::make_unique<QubitType>(basis_in_type.dim->copy())),
                                                      std::move(std::make_unique<QubitType>(basis_out_type.dim->copy())),
                                                      /*is_rev=*/true));
    trans.basis_in = std::move(basis_in);
    trans.basis_out = std::move(basis_out);
    return true;
}

bool QpuTypeCheckVisitor::isQfunc(const Type *type, DebugInfo &dbg, bool must_be_rev,
                                  DimVarExpr **dim_out) {
    const FuncType *func_type;
    if ((func_type = dynamic_cast<const FuncType *>(type))) {
        const QubitType *qubits_in, *qubits_out;
        if (!(qubits_in = dynamic_cast<const QubitType *>(func_type->lhs.get()))
            || !(qubits_out = dynamic_cast<const QubitType *>(func_type->rhs.get()))
            || *qubits_in->dim != *qubits_out->dim) {
            std::string desired_type = must_be_rev? "rev_qfunc[N]"
                                                  : "qfunc[N]";
            throw TypeException("A " + desired_type + " must be a function "
                                "from qubits to qubits, where the number of "
                                "qubits match. But got "
                                + func_type->toString(),
                                std::move(dbg.copy()));
        }
        if (must_be_rev && !func_type->is_rev) {
            throw TypeException("A rev_qfunc[N] must be reversible, but "
                                + func_type->toString() + " is not",
                                std::move(dbg.copy()));
        }
        if (dim_out) {
            *dim_out = qubits_in->dim.get();
        }
        return true;
    } else {
        return false;
    }
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Measure &meas) {
    meas.basis->visit(ctx, *this);
    meas.basis = std::move(wrapBasis(ctx, std::move(meas.basis),
                                     /*allow_empty=*/false));
    // Guaranteed by wrapBasis()
    const BasisType &basis_type = static_cast<const BasisType &>(meas.basis->getType());

    if (!basis_type.span.fullySpans()) {
        throw TypeException("The basis operand of a .measure must span the "
                            "full N-qubit space, but the basis provided does "
                            "not",
                            meas.basis->dbg->copy());
    }

    meas.type = std::make_unique<FuncType>(
        std::make_unique<QubitType>(std::move(basis_type.dim->copy())),
        std::make_unique<BitType>(std::move(basis_type.dim->copy())),
        /*is_rev=*/false);
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Project &proj) {
    proj.basis->visit(ctx, *this);
    proj.basis = std::move(wrapBasis(ctx, std::move(proj.basis),
                                     /*allow_empty=*/false));
    // Guaranteed by wrapBasis()
    const BasisType &basis_type = static_cast<const BasisType &>(proj.basis->getType());
    proj.type = std::make_unique<FuncType>(
        std::make_unique<QubitType>(std::move(basis_type.dim->copy())),
        std::make_unique<QubitType>(std::move(basis_type.dim->copy())),
        /*is_rev=*/false);
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Flip &flip) {
    flip.basis->visit(ctx, *this);
    flip.basis = std::move(wrapBasis(ctx, std::move(flip.basis),
                                     /*allow_empty=*/false));
    // Guaranteed by wrapBasis()
    const BasisType &basis_type = static_cast<const BasisType &>(flip.basis->getType());
    assert(basis_type.dim->isConstant() && "dimvars snuck into flip lowering");

    if (basis_type.dim->offset != 1) {
        throw TypeException("Flip takes only a 1-qubit basis, not a "
                            + std::to_string(basis_type.dim->offset)
                            + "-qubit basis",
                            std::move(flip.dbg->copy()));
    }

    if (!basis_type.span.fullySpans()) {
        throw TypeException("Flip requires a basis that fully spans",
                            std::move(flip.dbg->copy()));
    }

    flip.type = std::move(std::make_unique<FuncType>(
        std::move(std::make_unique<QubitType>(std::move(basis_type.dim->copy()))),
        std::move(std::make_unique<QubitType>(std::move(basis_type.dim->copy()))),
        /*is_rev=*/true));
    return true;
}

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Rotate &rot) {
    rot.theta->visit(ctx, *this);
    rot.basis->visit(ctx, *this);
    if (!dynamic_cast<const FloatType *>(&rot.theta->getType())) {
        throw TypeException("Expected Float in rotation, not " +
                            rot.theta->getType().toString(),
                            std::move(rot.dbg->copy()));
    }

    rot.basis = std::move(wrapBasis(ctx, std::move(rot.basis),
                                    /*allow_empty=*/false));
    // Guaranteed by wrapBasis()
    const BasisType &basis_type = static_cast<const BasisType &>(rot.basis->getType());
    assert(basis_type.dim->isConstant() && "dimvars snuck into flip lowering");

    if (basis_type.dim->offset != 1) {
        throw TypeException("Rotate takes only a 1-qubit basis, not a "
                            + std::to_string(basis_type.dim->offset)
                            + "-qubit basis",
                            std::move(rot.dbg->copy()));
    }

    if (!basis_type.span.fullySpans()) {
        throw TypeException("Rotate requires a basis that fully spans",
                            std::move(rot.dbg->copy()));
    }

    rot.type = std::make_unique<FuncType>(
        std::move(std::make_unique<QubitType>(std::move(basis_type.dim->copy()))),
        std::move(std::make_unique<QubitType>(std::move(basis_type.dim->copy()))),
        /*is_rev=*/true);
    return true;
}

// Nothing to check here
bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, Discard &discard) { return true; }

bool QpuTypeCheckVisitor::visit(ASTVisitContext &ctx, BasisLiteral &lit) {
    const DimVarExpr *first_dim = nullptr;
    std::unique_ptr<VeclistSpan> span = std::make_unique<VeclistSpan>((PrimitiveBasis)-1);

    if (lit.elts.empty()) {
        throw TypeException("Basis literal cannot be empty",
                            std::move(lit.dbg->copy()));
    }

    for (size_t i = 0; i < lit.elts.size(); i++) {
        ASTNode *elt = lit.elts[i].get();
        elt->visit(ctx, *this);

        if (!onlyLiterals(elt)) {
            throw TypeException("Elements of basis literals must be qubit "
                                "literals, but the element at index "
                                + std::to_string(i) + " is not",
                                std::move(elt->dbg->copy()));
        }

        const QubitType *elt_type = dynamic_cast<const QubitType *>(&elt->getType());
        if (!elt_type) {
            throw TypeException("Elements of basis literals must have type "
                                "Qubit[N], but the element at index "
                                + std::to_string(i) + " has type "
                                + elt->getType().toString() + " instead",
                                std::move(elt->dbg->copy()));
        }

        const DimVarExpr *this_dim = elt_type->dim.get();
        if (!first_dim) {
            first_dim = this_dim;
        } else if (*this_dim != *first_dim) {
            throw TypeException("All qubit literals in a basis literal must "
                                "have the same number of qubits, yet first "
                                "state has dimension " + first_dim->toString() +
                                " but index " + std::to_string(i)
                                + " has dimension " +
                                this_dim->toString(),
                                std::move(elt->dbg->copy()));
        }

        PrimitiveBasis this_prim_basis = basisVectorPrimitiveBasisOrError(elt, i);
        assert(this_prim_basis != (PrimitiveBasis)-1 && "Unit literal () snuck into basis "
                                          "literal typechecking?");
        if (span->prim_basis == (PrimitiveBasis)-1) {
            span->prim_basis = this_prim_basis;
        } else if (span->prim_basis != this_prim_basis) {
            throw TypeException("States in a basis literal must consist of "
                                "vectors only from std[N], or from only ij[N] "
                                "instead, or from only pm[N] instead. Yet "
                                "the first vector is from "
                                + prim_basis_name(span->prim_basis) + "[N] "
                                "and index " + std::to_string(i) + " is from "
                                + prim_basis_name(this_prim_basis) + "[N]",
                                std::move(elt->dbg->copy()));
        }

        llvm::APInt eigenbits = basisVectorEigenbitsOrError(elt, i);
        if (eigenbits.getBitWidth() == 0) {
            throw TypeException("Elements of basis literals cannot be empty "
                                "yet the element at index "
                                + std::to_string(i) + " is",
                                std::move(elt->dbg->copy()));
        }
        if (!span->vecs.insert(eigenbits).second) {
            throw TypeException("Elements of basis literals must be distinct "
                                "(excluding phases), but the element at index "
                                + std::to_string(i) + " is a repeat",
                                std::move(elt->dbg->copy()));
        }
    }

    std::unique_ptr<BasisType> basis_type = std::make_unique<BasisType>(std::move(first_dim->copy()));
    basis_type->span.clear();
    basis_type->span.append(std::move(span));
    lit.type = std::move(basis_type);
    return true;
}

///////////////////////// @CLASSICAL TYPE CHECKING /////////////////////////

bool ClassicalTypeCheckVisitor::visitNonClassicalNode(ASTVisitContext &ctx, ASTNode &node) {
    throw TypeException("How is there a non-@classical AST node "
                        + node.label() + " in a @classical AST?",
                        std::move(node.dbg->copy()));
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, BitUnaryOp &unOp) {
    unOp.operand->visit(ctx, *this);
    const Type &operandType = unOp.operand->getType();
    if (!dynamic_cast<const BitType *>(&operandType)) {
        throw TypeException("Unary bit operation operand must be performed "
                            "on bits, not " + operandType.toString(),
                            std::move(unOp.dbg->copy()));
    }
    unOp.type = std::move(operandType.copy());
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, BitBinaryOp &binOp) {
    binOp.left->visit(ctx, *this);
    binOp.right->visit(ctx, *this);
    const Type &leftType = binOp.left->getType();
    const Type &rightType = binOp.right->getType();

    if (bit_op_is_broadcast(binOp.op) && !(leftType <= rightType)
            && !(rightType <= leftType)) {
        throw TypeException("Binary operation operands do not match: "
                            + leftType.toString() + " differs from "
                            + rightType.toString(),
                            std::move(binOp.dbg->copy()));
    } else if (!dynamic_cast<const BitType *>(&leftType)
               || !dynamic_cast<const BitType *>(&rightType)) {
        throw TypeException("Binary bit operation operand must be performed "
                            "on bits, not " + leftType.toString() + " and "
                            + rightType.toString(),
                            std::move(binOp.dbg->copy()));
    }

    if (leftType <= rightType) {
        binOp.type = std::move(rightType.copy());
    } else { // rightType <= leftType
        binOp.type = std::move(leftType.copy());
    }
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, BitReduceOp &reduce) {
    reduce.operand->visit(ctx, *this);
    const Type &operandType = reduce.operand->getType();
    if (!dynamic_cast<const BitType *>(&operandType)) {
        throw TypeException("Bit reduction operation must be performed "
                            "on bits, not " + operandType.toString(),
                            std::move(reduce.dbg->copy()));
    }
    reduce.type = std::move(std::make_unique<BitType>(
            std::move(std::make_unique<DimVarExpr>("", 1))));
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, BitConcat &concat) {
    concat.left->visit(ctx, *this);
    concat.right->visit(ctx, *this);
    const BitType *leftType, *rightType;
    if (!(leftType = dynamic_cast<const BitType *>(&concat.left->getType()))
            || !(rightType = dynamic_cast<const BitType *>(&concat.right->getType()))) {
        throw TypeException("Bit concat operation must be performed on bits, not "
                            + concat.left->getType().toString()
                            + " and " + concat.right->getType().toString(),
                            std::move(concat.dbg->copy()));
    }
    assert(leftType->dim->isConstant() && rightType->dim->isConstant()
           && "dimvars snuck into bit concat type checking");

    DimVarValue sum_dim = leftType->dim->offset + rightType->dim->offset;
    concat.type = std::move(std::make_unique<BitType>(
            std::move(std::make_unique<DimVarExpr>("", sum_dim))));
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, BitRepeat &repeat) {
    repeat.bits->visit(ctx, *this);
    const BitType *operandType;
    if (!(operandType = dynamic_cast<const BitType *>(&repeat.bits->getType()))) {
        throw TypeException("Bit repeat operation must be performed on bits, not "
                            + repeat.bits->getType().toString(),
                            std::move(repeat.dbg->copy()));
    }
    assert(operandType->dim->isConstant()
           && repeat.amt->isConstant()
           && "dimvars snuck into bit repeat type checking");

    DimVarValue new_dim = operandType->dim->offset * repeat.amt->offset;
    repeat.type = std::move(std::make_unique<BitType>(
            std::make_unique<DimVarExpr>("", new_dim)));
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, ModMulOp &mul) {
    mul.y->visit(ctx, *this);
    const Type &y_type = mul.y->getType();
    if (!dynamic_cast<const BitType *>(&y_type)) {
        throw TypeException("Modular multiplication must be performed "
                            "on bits, not " + y_type.toString(),
                            std::move(mul.dbg->copy()));
    }
    if (!mul.x->isConstant() || mul.x->offset <= 0) {
        throw TypeException("x for modular multiplication is invalid. Expected "
                            "a positive integer but got " + mul.x->toString(),
                            std::move(mul.dbg->copy()));
    }
    if (!mul.j->isConstant() || mul.j->offset < 0) {
        throw TypeException("j in 2**j for modular multiplication is invalid. "
                            "Expected a non-negative integer but got "
                            + mul.j->toString(),
                            std::move(mul.dbg->copy()));
    }
    if (!mul.modN->isConstant() || mul.modN->offset < 2) {
        throw TypeException("the N in ...% N for modular multiplication is "
                            "invalid. Expected an integer >= 2 but got "
                            + mul.modN->toString(),
                            std::move(mul.dbg->copy()));
    }
    mul.type = std::move(y_type.copy());
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, BitLiteral &bitLit) {
    if (!bitLit.val->isConstant() || bitLit.val->offset < 0) {
        throw TypeException("Value V of bit literal bit(V, N) is invalid. "
                            "Expected an integer >= 0 but got "
                            + bitLit.val->toString(),
                            std::move(bitLit.dbg->copy()));
    }
    if (!bitLit.n_bits->isConstant() || bitLit.n_bits->offset < 1) {
        throw TypeException("Number of bits N of bit literal bit(V, N) is invalid. "
                            "Expected an integer >= 1 but got "
                            + bitLit.n_bits->toString(),
                            std::move(bitLit.dbg->copy()));
    }

    DimVarValue n_bits_needed = !bitLit.val->offset ? 0
                                                     : bits_needed(bitLit.val->offset);
    if (bitLit.n_bits->offset < n_bits_needed) {
        throw TypeException("Number of bits N=" + bitLit.n_bits->toString()
                            + " of bit literal bit(V, N) is less than the "
                            + "number of bits required to represent V, "
                            + std::to_string(n_bits_needed),
                            std::move(bitLit.dbg->copy()));
    }

    bitLit.type = std::make_unique<BitType>(bitLit.n_bits->copy());
    return true;
}

bool ClassicalTypeCheckVisitor::visit(ASTVisitContext &ctx, Slice &slice) {
    slice.val->visit(ctx, *this);
    const BitType *bit_type;
    if (!(bit_type = dynamic_cast<const BitType *>(&slice.val->getType()))) {
        throw TypeException("Accessing individual bits with b[...] is allowed "
                            "only if b is a bit[N]. Here it is a "
                            + slice.val->getType().toString() + " instead.",
                            std::move(slice.dbg->copy()));
    }
    assert(bit_type->dim->isConstant() && "dimvars snuck into type of bit!");

    DimVarValue n_bits, start_idx;
    validateSlice(slice, bit_type->dim->offset, n_bits, start_idx);
    slice.type = std::make_unique<BitType>(std::make_unique<DimVarExpr>("", n_bits));
    return true;
}

