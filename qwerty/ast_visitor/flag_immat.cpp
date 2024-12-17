#include "ast.hpp"
#include "ast_visitor.hpp"

bool FlagImmaterialVisitor::visitNode(ASTVisitContext &ctx, ASTNode &node) {
    Kernel *kern;
    if ((kern = dynamic_cast<Kernel *>(&node))
            && (kern->type && !kern->type->isMaterializable())) {
        throw TypeException("A kernel type cannot include immaterial types "
                            "(or a type containing an immaterial type). A "
                            "basis is an example of an immaterial type. "
                            "Offending kernel type: " +
                            (kern->type? kern->type->toString()
                                       : std::string("(unset)")),
                            std::move(node.dbg->copy()));
    }

    if (!node.hasType() || node.getType().isMaterializable()) {
        // Nothing to check here.
        return true;
    }

    // Henceforth, we are going to throw a type error for this node unless we
    // can prove it is among our supported cases

    // Because parent=node is passed for the root node, which we don't want to
    // bother getting upset about if this visitor is run on code typed into a
    // REPL and a basis is intentionally the root node of the AST.
    ASTNode *parent = &ctx.parent == &node ? nullptr : &ctx.parent;

    if (dynamic_cast<const BasisType *>(&node.getType())) {
        Pred *pred;
        Rotate *rot;
        // This is basically a blacklist of nodes from having a basis type
        // and a whitelist for the parent–child edges in which a basis
        // child is legal
        if (!dynamic_cast<Variable *>(&node)
                && !dynamic_cast<Instantiate *>(&node)
                && (!parent
                    || dynamic_cast<Prepare *>(parent)
                    || ((pred = dynamic_cast<Pred *>(parent))
                        && (pred->order == PRED_ORDER_UNKNOWN
                            || pred->basis.get() == &node))
                    || dynamic_cast<BiTensor *>(parent)
                    || dynamic_cast<BroadcastTensor *>(parent)
                    || dynamic_cast<BasisTranslation *>(parent)
                    || dynamic_cast<Measure *>(parent)
                    || dynamic_cast<Project *>(parent)
                    || dynamic_cast<Flip *>(parent)
                    || ((rot = dynamic_cast<Rotate *>(parent))
                        && rot->basis.get() == &node))) {
            // All good! :)
            return true;
        }
    } else if (node.getType().collapseToHomogeneousArray<ComplexType>()) {
        if (dynamic_cast<Lift *>(parent)) {
            // :)
            return true;
        }
    }

    throw TypeException("Type that includes an immaterial types "
                        "used improperly. Offending type: "
                        + (node.hasType()? node.getType().toString()
                                         : std::string("(unset)")),
                        std::move(node.dbg->copy()));
    return true;
}
