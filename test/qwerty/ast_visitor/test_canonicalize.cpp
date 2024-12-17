#include "gtest/gtest.h"
#include "test_support.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"

// Useful incantation for debugging:
// a TODO is to get gtest to print this for us
//{
//    GraphvizVisitor visitor;
//    ast.walk(visitor);
//    std::cerr << "actual\n" << visitor.str() << "\n" << std::endl;
//}
//{
//    GraphvizVisitor visitor;
//    expected_ast.walk(visitor);
//    std::cerr << "expected\n" << visitor.str() << "\n" << std::endl;
//}

TEST(canonicalize, canonBroadcastMeasure) {
    Return ast(mock_dbg(1),
        std::make_unique<BroadcastTensor>(
            mock_dbg(2),
            std::make_unique<Measure>(mock_dbg(3),
                std::make_unique<BuiltinBasis>(mock_dbg(4),
                    Z,
                    mock_dimvar_expr(2))),
            mock_dimvar_expr(3)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<Measure>(mock_dbg(3),
            std::make_unique<BuiltinBasis>(mock_dbg(4),
                Z,
                mock_dimvar_expr(6))));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonFloatDimVarExpr) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatDimVarExpr>(
            mock_dbg(2), mock_dimvar_expr(37)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<FloatLiteral>(
            mock_dbg(2), 37.0));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonAdjAdj) {
    Return ast(mock_dbg(1),
        std::make_unique<Adjoint>(mock_dbg(2),
            std::make_unique<Adjoint>(mock_dbg(3),
                std::make_unique<Variable>(mock_dbg(4), "x"))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<Variable>(mock_dbg(4), "x"));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonAdjFlip) {
    Return ast(mock_dbg(1),
        std::make_unique<Adjoint>(mock_dbg(2),
            std::make_unique<Flip>(mock_dbg(3),
                std::make_unique<BuiltinBasis>(mock_dbg(4),
                    Z, mock_dimvar_expr(1)))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<Flip>(mock_dbg(3),
            std::make_unique<BuiltinBasis>(mock_dbg(4),
                Z, mock_dimvar_expr(1))));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonAdjId) {
    Return ast(mock_dbg(1),
        std::make_unique<Adjoint>(mock_dbg(2),
                std::make_unique<Identity>(mock_dbg(3),
                    mock_dimvar_expr(42))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
            std::make_unique<Identity>(mock_dbg(3),
                mock_dimvar_expr(42)));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonAdjRotate) {
    Return ast(mock_dbg(1),
        std::make_unique<Adjoint>(mock_dbg(2),
            std::make_unique<Rotate>(mock_dbg(3),
                std::make_unique<BuiltinBasis>(mock_dbg(4),
                    Z, mock_dimvar_expr(1)),
                std::make_unique<FloatLiteral>(mock_dbg(5),
                    3.14159265))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<Rotate>(mock_dbg(3),
            std::make_unique<BuiltinBasis>(mock_dbg(4),
                Z, mock_dimvar_expr(1)),
            std::make_unique<FloatLiteral>(mock_dbg(5),
                -3.14159265)));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonAdjBtrans) {
    Return ast(mock_dbg(1),
        std::make_unique<Adjoint>(mock_dbg(2),
            std::make_unique<BasisTranslation>(mock_dbg(3),
                std::make_unique<BuiltinBasis>(mock_dbg(4),
                    Z, mock_dimvar_expr(4)),
                std::make_unique<BuiltinBasis>(mock_dbg(5),
                    X, mock_dimvar_expr(4)))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<BasisTranslation>(mock_dbg(3),
            std::make_unique<BuiltinBasis>(mock_dbg(5),
                X, mock_dimvar_expr(4)),
            std::make_unique<BuiltinBasis>(mock_dbg(4),
                Z, mock_dimvar_expr(4))));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonPredBtrans) {
    std::vector<std::unique_ptr<ASTNode>> blit;
    blit.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(4),
            MINUS,
            Z,
            mock_dimvar_expr(3)));

    Return ast(mock_dbg(1),
        std::make_unique<Pred>(mock_dbg(2),
            PRED_ORDER_B_U,
            std::make_unique<BasisLiteral>(mock_dbg(3),
                std::move(blit)),
            std::make_unique<BasisTranslation>(mock_dbg(4),
                std::make_unique<BuiltinBasis>(mock_dbg(5),
                    Z, mock_dimvar_expr(4)),
                std::make_unique<BuiltinBasis>(mock_dbg(6),
                    X, mock_dimvar_expr(4)))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    std::vector<std::unique_ptr<ASTNode>> blit1;
    blit1.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(4),
            MINUS,
            Z,
            mock_dimvar_expr(3)));
    std::vector<std::unique_ptr<ASTNode>> blit2;
    blit2.push_back(
        std::make_unique<QubitLiteral>(mock_dbg(4),
            MINUS,
            Z,
            mock_dimvar_expr(3)));
    Return expected_ast(mock_dbg(1),
        std::make_unique<BasisTranslation>(mock_dbg(4),
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<BasisLiteral>(mock_dbg(3),
                    std::move(blit1)),
                std::make_unique<BuiltinBasis>(mock_dbg(5),
                    Z, mock_dimvar_expr(4))),
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<BasisLiteral>(mock_dbg(3),
                    std::move(blit2)),
                std::make_unique<BuiltinBasis>(mock_dbg(6),
                    X, mock_dimvar_expr(4)))));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonPredBtransFullySpan) {
    std::unique_ptr<BuiltinBasis> std3 =
        std::make_unique<BuiltinBasis>(mock_dbg(3),
            Y, mock_dimvar_expr(3));
    std3->type.span.append(std::make_unique<FullSpan>(3));

    Return ast(mock_dbg(1),
        std::make_unique<Pred>(mock_dbg(2),
            PRED_ORDER_B_U,
            std::move(std3),
            std::make_unique<BasisTranslation>(mock_dbg(4),
                std::make_unique<BuiltinBasis>(mock_dbg(5),
                    Z, mock_dimvar_expr(4)),
                std::make_unique<BuiltinBasis>(mock_dbg(6),
                    X, mock_dimvar_expr(4)))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<BiTensor>(mock_dbg(2),
            std::make_unique<Identity>(mock_dbg(2),
                mock_dimvar_expr(3)),
            std::make_unique<BasisTranslation>(mock_dbg(4),
                std::make_unique<BuiltinBasis>(mock_dbg(5),
                    Z, mock_dimvar_expr(4)),
                std::make_unique<BuiltinBasis>(mock_dbg(6),
                    X, mock_dimvar_expr(4)))));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonNegNegFloat) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatNeg>(mock_dbg(2),
            std::make_unique<FloatNeg>(mock_dbg(3),
                std::make_unique<Variable>(mock_dbg(4),
                    "f"))));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<Variable>(mock_dbg(4),
            "f"));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonNegFloatLiteral) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatNeg>(mock_dbg(2),
            std::make_unique<FloatLiteral>(mock_dbg(3),
                1.57079)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
            std::make_unique<FloatLiteral>(mock_dbg(3),
                -1.57079));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonFloatDivLiterals) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_DIV,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                16.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                2.0)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<FloatLiteral>(mock_dbg(2),
            8.0));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonFloatDivLiteralsSkipDivByZero) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_DIV,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                32.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                0.0)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    // Unchanged
    Return expected_ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_DIV,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                32.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                0.0)));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonFloatPowLiterals) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_POW,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                2.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                4.0)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<FloatLiteral>(mock_dbg(2),
            16.0));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonFloatPowLiteralsSkipInsanity) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_POW,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                2.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                99999999.0)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    // Unchanged
    Return expected_ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_POW,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                2.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                99999999.0)));

    EXPECT_EQ(ast, expected_ast);
}

TEST(canonicalize, canonFloatMulLiterals) {
    Return ast(mock_dbg(1),
        std::make_unique<FloatBinaryOp>(mock_dbg(2),
            FLOAT_MUL,
            std::make_unique<FloatLiteral>(mock_dbg(3),
                2.0),
            std::make_unique<FloatLiteral>(mock_dbg(4),
                3.0)));

    CanonicalizeVisitor visitor;
    ast.walk(visitor);

    Return expected_ast(mock_dbg(1),
        std::make_unique<FloatLiteral>(mock_dbg(2),
            6.0));

    EXPECT_EQ(ast, expected_ast);
}
