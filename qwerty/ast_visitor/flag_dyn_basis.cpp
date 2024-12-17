#include "ast.hpp"
#include "ast_visitor.hpp"

bool FlagDynamicBasisVisitor::visitNode(ASTVisitContext &ctx, ASTNode &node) {
    Kernel *kern;
    if ((kern = dynamic_cast<Kernel *>(&node))
            && ((kern->type && kern->type->contains<BasisType>())
                || std::any_of(kern->capture_types.begin(),
                               kern->capture_types.end(),
                               [](std::unique_ptr<Type> &ty) {
                                   return ty->contains<BasisType>();
                               }))) {
        throw TypeException("A kernel cannot contain a basis in any of "
                            "its type metadata",
                            std::move(node.dbg->copy()));
    }

    if (!node.hasType()
            || !node.getType().contains<BasisType>()) {
        // Nothing to check here.
        return true;
    }

    if (dynamic_cast<Variable *>(&node)
            || dynamic_cast<Instantiate *>(&node)
            || dynamic_cast<Kernel *>(&node)) {
        throw TypeException("The AST node " + node.label() + " cannot have "
                            "type basis",
                            std::move(node.dbg->copy()));
    }

    // Because parent=node is passed for the root node, which we don't want to
    // bother getting upset about if this visitor is run on code typed into a
    // REPL and a basis is intentionally the root node of the AST.
    if (&ctx.parent == &node) {
        return true;
    }

    ASTNode *parent = &ctx.parent;
    Pred *pred;
    Rotate *rot;
    // This is basically a whitelist of the parent–child edges in which a basis
    // child is legal
    if (!dynamic_cast<Prepare *>(parent)
            && (!(pred = dynamic_cast<Pred *>(parent))
                || (pred->order != PRED_ORDER_UNKNOWN
                    && pred->body.get() == &node))
            && !dynamic_cast<BiTensor *>(parent)
            && !dynamic_cast<BroadcastTensor *>(parent)
            && !dynamic_cast<BasisTranslation *>(parent)
            && !dynamic_cast<Measure *>(parent)
            && !dynamic_cast<Project *>(parent)
            && !dynamic_cast<Flip *>(parent)
            && (!(rot = dynamic_cast<Rotate *>(parent)) || rot->theta.get() == &node)) {
        throw TypeException("A basis cannot be used in an expression",
                            std::move(node.dbg->copy()));
    }

    return true;
}
