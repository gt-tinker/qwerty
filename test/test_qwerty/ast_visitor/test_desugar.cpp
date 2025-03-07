#include "gtest/gtest.h"
#include "test_support.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"

TEST(desugar, desugarBasisTranslation) {
    std::vector<std::unique_ptr<ASTNode>> mapping1{};
    mapping1.push_back(
        std::make_unique<BasisTranslation>(
            mock_dbg(1),
            std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, X, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, Z, mock_dimvar_expr(1))
        )
    );
    mapping1.push_back(
        std::make_unique<BasisTranslation>(
            mock_dbg(1),
            std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, X, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, Z, mock_dimvar_expr(1))
        )
    );

    Return ast(
        mock_dbg(1),
        std::make_unique<BasisLiteral>(
            mock_dbg(1),
            std::move(mapping1)
        )
    );

    std::vector<std::unique_ptr<ASTNode>> mapping2lhs_elts{};
    mapping2lhs_elts.push_back(std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, X, mock_dimvar_expr(1)));
    mapping2lhs_elts.push_back(std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, X, mock_dimvar_expr(1)));

    std::unique_ptr<ASTNode> mapping2lhs = std::make_unique<BasisLiteral>(
        mock_dbg(1),
        std::move(mapping2lhs_elts)
    );

    std::vector<std::unique_ptr<ASTNode>> mapping2rhs_elts{};
    mapping2rhs_elts.push_back(std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, Z, mock_dimvar_expr(1)));
    mapping2rhs_elts.push_back(std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, Z, mock_dimvar_expr(1)));

    std::unique_ptr<ASTNode> mapping2rhs = std::make_unique<BasisLiteral>(
        mock_dbg(1),
        std::move(mapping2rhs_elts)
    );

    Return expected_ast(
        mock_dbg(1),
        std::make_unique<BasisTranslation>(
            mock_dbg(1),
            std::move(mapping2lhs),
            std::move(mapping2rhs)
        )
    );

    DesugarVisitor visitor;
    ast.walk(visitor);

    EXPECT_EQ(ast, expected_ast);
}

// When this process doesn't find a perfect match, it simply leaves the tree
// untransformed. Typechecking already provides the infrastructure to validate
// if the basis is well-formed.
TEST(desugar, ignoreNormalLiteral) {
    std::vector<std::unique_ptr<ASTNode>> qubits{};
    qubits.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, X, mock_dimvar_expr(1))
    );

    qubits.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, X, mock_dimvar_expr(1))
    );

    BasisLiteral ast(mock_dbg(1), std::move(qubits));

    // Apparently "ast->copy" changes some metadata on the new tree, so this
    // is how we compare two ASTs using GTest...
    std::vector<std::unique_ptr<ASTNode>> qubits_exp{};
    qubits_exp.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, X, mock_dimvar_expr(1))
    );

    qubits_exp.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, X, mock_dimvar_expr(1))
    );

    BasisLiteral expected_ast(mock_dbg(1), std::move(qubits_exp));

    DesugarVisitor visitor;
    ast.walk(visitor);

    EXPECT_EQ(ast, expected_ast);
}

TEST(desugar, ignoreMalformedTranslation) {
    std::vector<std::unique_ptr<ASTNode>> bad_elts{};

    bad_elts.push_back(
        std::make_unique<BasisTranslation>(
            mock_dbg(1),
            std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, X, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, Z, mock_dimvar_expr(1))
        )
    );

    bad_elts.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, X, mock_dimvar_expr(1))
    );

    BasisLiteral ast(mock_dbg(1), std::move(bad_elts));

    // Want it unchanged.
    std::vector<std::unique_ptr<ASTNode>> bad_elts_exp{};
    bad_elts_exp.push_back(
        std::make_unique<BasisTranslation>(
            mock_dbg(1),
            std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, X, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(1), PLUS, Z, mock_dimvar_expr(1))
        )
    );

    bad_elts_exp.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(1), MINUS, X, mock_dimvar_expr(1))
    );

    BasisLiteral expected_ast(mock_dbg(1), std::move(bad_elts_exp));

    DesugarVisitor visitor;
    ast.walk(visitor);

    EXPECT_EQ(ast, expected_ast);
}
