#include "ast.hpp"
#include "ast_visitor.hpp"

void GraphvizVisitor::init(ASTNode &root) {
    next_node_index = 0;
    ss << "digraph {\n"
       << "ordering=out\n"
       << "dpi=200\n";
    // Create root node
    getNodeIndex(&root);
}

bool GraphvizVisitor::visitNode(ASTVisitContext &ctx, ASTNode &node) {
    // Need this check because parent=node is passed for the root node, and we
    // don't need a spurious reflexive edge on the root node
    if (&ctx.parent != &node) {
        drawEdge(&ctx.parent, &node, ctx.label);
    }
    return true;
}

void GraphvizVisitor::finish() {
    ss << "}\n";
}

std::string GraphvizVisitor::str() {
    return ss.str();
}

size_t GraphvizVisitor::getNodeIndex(ASTNode *node) {
    auto pair = node_indices.emplace(node, next_node_index);
    if (pair.second) {
        ss << "node" << next_node_index << "[shape=box, label=\""
           << node->label();
        if (node->hasType()) {
            const Type &ty = node->getType();
            ss << "\\n Type: " << ty.toString();
            if (const BasisType *basis_ty = dynamic_cast<const BasisType *>(&ty)) {
                ss << "\\n Span: " << basis_ty->span;
            }
        }
        std::vector<std::string> additional_info = node->getAdditionalMetadata();
        for (std::string &info : additional_info) {
            ss << "\\n" << info;
        }
        ss << "\"]\n";
        next_node_index++;
    }
    // Best C++ notation
    return pair.first->second;
}

void GraphvizVisitor::drawEdge(ASTNode *from, ASTNode *to, std::string edge_label) {
    size_t from_idx = getNodeIndex(from);
    size_t to_idx = getNodeIndex(to);
    ss << "node" << from_idx << " -> node" << to_idx << " [label=\"" << edge_label << "\"]\n";
}
