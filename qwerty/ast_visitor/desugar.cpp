#include "ast.hpp"
#include "ast_visitor.hpp"

// Rewrites basis literals of the form {a >> b, c >> d, ...} into {a, c, ...} >> {b, d, ...}
// I'm open to treating this as "canonicalization" if you don't mind it running
// *before* typechecking, but my intuition tells me it's a bad idea.
bool DesugarVisitor::visit(ASTVisitContext &ctx, BasisLiteral &lit) {
    for (auto const &elt : lit.elts) {
        // Our types don't implement this.
        if (!dynamic_cast<BasisTranslation *>(elt.get())) {
            // Forward this to type-checking so it can admit literals 
            // of the form {'qbl1', 'qbl2', ...} or complain appropriately
            return true;
        }
    }

    std::vector<std::unique_ptr<ASTNode>> input_basis{};
    input_basis.reserve(lit.elts.size());
    std::vector<std::unique_ptr<ASTNode>> output_basis{};
    output_basis.reserve(lit.elts.size());

    for (auto const &elt : lit.elts) {
        // Already know it's all BasisTranslation, static cast is safe.
        BasisTranslation *btrans = static_cast<BasisTranslation *>(elt.get());
        input_basis.push_back(std::move(btrans->basis_in));
        output_basis.push_back(std::move(btrans->basis_out));
    }

    // Move everything everywhere everywhen
    // FIXME: Don't know how we handle debug info...
    std::unique_ptr<BasisTranslation> new_btrans =
        std::make_unique<BasisTranslation>(std::move(lit.dbg->copy()),
                                           std::make_unique<BasisLiteral>(std::move(lit.dbg->copy()),
                                                                          std::move(input_basis)),
                                           std::make_unique<BasisLiteral>(std::move(lit.dbg->copy()),
                                                                          std::move(output_basis)));

    ctx.ptr = std::move(new_btrans);
    return false;
}
