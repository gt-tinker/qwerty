// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include "gtest/gtest.h"
#include "test_support.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"

////////////////////////////////////////////////////////////////////////
////////////////////////// @QPU TYPE CHECKING //////////////////////////
////////////////////////////////////////////////////////////////////////

TEST(qpuTypeChecking, classicalNode) {
    BitUnaryOp ast(mock_dbg(1), BIT_NOT,
        std::make_unique<BitLiteral>(mock_dbg(2),
            mock_dimvar_expr(0b1101), mock_dimvar_expr(4)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "How is there a non-@qpu AST node BitUnaryOp in a @qpu AST?");
}

TEST(qpuTypeChecking, noShadowingAllowed) {
    Assign first_assign(mock_dbg(1),
                        "q",
                        std::make_unique<QubitLiteral>(mock_dbg(2),
                                                       PLUS, Z,
                                                       mock_dimvar_expr(2)));
    Assign second_assign(mock_dbg(3),
                         "q",
                         std::make_unique<QubitLiteral>(mock_dbg(4),
                                                      MINUS, X,
                                                      mock_dimvar_expr(2)));

    QpuTypeCheckVisitor visitor;
    first_assign.walk(visitor);

    QW_EXPECT_THROW_MSG({
        second_assign.walk(visitor);
    }, TypeException,
    "All variables are immutable and cannot be shadowed: Variable q was already defined.");
}

////////////////////////// SLICE //////////////////////////

// (1.0, 2.0, 3.0)[[0:3]]
TEST(qpuTypeCheckingSlice, sliceFloatTupleFull) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/mock_dimvar_expr(0),
        /*upper=*/mock_dimvar_expr(3));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    TupleType expected_type(
        unique_vec<Type>(
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>()));
    EXPECT_EQ(*ast.type, expected_type);
}

// (1.0, 2.0, 3.0)[[0:]]
TEST(qpuTypeCheckingSlice, sliceFloatTupleFullNoUpper) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/mock_dimvar_expr(0),
        /*upper=*/nullptr);

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    TupleType expected_type(
        unique_vec<Type>(
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>()));
    EXPECT_EQ(*ast.type, expected_type);
}

// (1.0, 2.0, 3.0)[[:3]]
TEST(qpuTypeCheckingSlice, sliceFloatTupleFullNoLower) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/nullptr,
        /*upper=*/mock_dimvar_expr(3));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    TupleType expected_type(
        unique_vec<Type>(
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>()));
    EXPECT_EQ(*ast.type, expected_type);
}

// (1.0, 2.0, 3.0)[[:]]
TEST(qpuTypeCheckingSlice, sliceFloatTupleFullNeitherLowerNorUpper) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/nullptr,
        /*upper=*/nullptr);

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    TupleType expected_type(
        unique_vec<Type>(
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>()));
    EXPECT_EQ(*ast.type, expected_type);
}

// (1.0, 2.0, 3.0)[[2:]]
TEST(qpuTypeCheckingSlice, sliceFloatTupleLast) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/mock_dimvar_expr(2),
        /*upper=*/nullptr);

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FloatType expected_type;
    EXPECT_EQ(*ast.type, expected_type);
}

// (1.0, 2.0, 3.0)[[:2]]
TEST(qpuTypeCheckingSlice, sliceFloatTupleFirstTwo) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/nullptr,
        /*upper=*/mock_dimvar_expr(2));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    TupleType expected_type(
        unique_vec<Type>(
            std::make_unique<FloatType>(),
            std::make_unique<FloatType>()));
    EXPECT_EQ(*ast.type, expected_type);
}

// 1.0[[0:1]]
TEST(qpuTypeCheckingSlice, sliceFloat) {
    Slice ast(mock_dbg(1),
        std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
        /*lower=*/mock_dimvar_expr(0),
        /*upper=*/mock_dimvar_expr(1));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FloatType expected_type;
    EXPECT_EQ(*ast.type, expected_type);
}

// '0000'[[:]]
TEST(qpuTypeCheckingSlice, quantumSlicing) {
    Slice ast(mock_dbg(1),
        std::make_unique<QubitLiteral>(mock_dbg(3), PLUS, Z,
                                       mock_dimvar_expr(4)),
        /*lower=*/nullptr,
        /*upper=*/nullptr);

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Currently, you can slice only classical types");
}

// (1.0, 2.0, 3.0)[[-1:]]
TEST(qpuTypeCheckingSlice, sliceNegativeLower) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/mock_dimvar_expr(-1),
        /*upper=*/nullptr);

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Slice index negative: -1 < 0");
}

// (1.0, 2.0, 3.0)[[:-1]]
TEST(qpuTypeCheckingSlice, sliceNegativeUpper) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/nullptr,
        /*upper=*/mock_dimvar_expr(-1));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Slice index negative: -1 < 0");
}

// (1.0, 2.0, 3.0)[[2:1]]
TEST(qpuTypeCheckingSlice, sliceLowerBeyondUpper) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/mock_dimvar_expr(2),
        /*upper=*/mock_dimvar_expr(1));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Slice lower bound larger than upper bound: 2 >= 1");
}

// (1.0, 2.0, 3.0)[[:4]]
TEST(qpuTypeCheckingSlice, sliceUpperOob) {
    Slice ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<FloatLiteral>(mock_dbg(3), 1.0),
                std::make_unique<FloatLiteral>(mock_dbg(4), 2.0),
                std::make_unique<FloatLiteral>(mock_dbg(5), 3.0))),
        /*lower=*/nullptr,
        /*upper=*/mock_dimvar_expr(4));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Slice index out of bounds: 4 > 3");
}

////////////////////////// ADJOINT //////////////////////////

TEST(qpuTypeCheckingAdjoint, valid) {
    Adjoint ast(mock_dbg(1),
        std::make_unique<Flip>(mock_dbg(2),
            std::make_unique<BuiltinBasis>(mock_dbg(3),
                Z, mock_dimvar_expr(1))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FuncType expected_type(
        std::make_unique<QubitType>(mock_dimvar_expr(1)),
        std::make_unique<QubitType>(mock_dimvar_expr(1)),
        /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}

TEST(qpuTypeCheckingAdjoint, nonFunc) {
    Adjoint ast(mock_dbg(1),
        std::make_unique<QubitLiteral>(mock_dbg(2),
            PLUS, Z, mock_dimvar_expr(2)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Operand f of adjoint ~f must be rev_qfunc[N] (a reversible function from qubit[N] to qubit[N]), not Qubit[2]");
}

TEST(qpuTypeCheckingAdjoint, nonRevFunc) {
    Adjoint ast(mock_dbg(1),
        std::make_unique<Variable>(mock_dbg(2), "func"));

    FuncType non_rev_type(
        std::make_unique<QubitType>(mock_dimvar_expr(2)),
        std::make_unique<QubitType>(mock_dimvar_expr(2)),
        /*is_rev=*/false);
    QpuTypeCheckVisitor visitor;
    visitor.variables.insert({"func", non_rev_type.copy()});

    QW_EXPECT_THROW_MSG({
        ast.walk(visitor);
    }, TypeException,
    "A rev_qfunc[N] must be reversible, but (Qubit[2]→Qubit[2]) is not");
}

TEST(qpuTypeCheckingAdjoint, mismatchedFunc) {
    Adjoint ast(mock_dbg(1),
        std::make_unique<Variable>(mock_dbg(2), "func"));

    FuncType mismatched_func(
        std::make_unique<QubitType>(mock_dimvar_expr(2)),
        std::make_unique<QubitType>(mock_dimvar_expr(3)),
        /*is_rev=*/true);
    QpuTypeCheckVisitor visitor;
    visitor.variables.insert({"func", mismatched_func.copy()});

    QW_EXPECT_THROW_MSG({
        ast.walk(visitor);
    }, TypeException,
    "A rev_qfunc[N] must be a function from qubits to qubits, where the number of qubits match. But got (Qubit[2]—ʳᵉᵛ→Qubit[3])");
}

////////////////////////// PREPARE //////////////////////////

// Given x = bit[4](0b1101),
// x.prep
TEST(qpuTypeCheckingPrepare, prepBits) {
    Prepare ast(mock_dbg(1),
        std::make_unique<Variable>(mock_dbg(2), "x"));

    BitType bit_type(mock_dimvar_expr(4));
    QpuTypeCheckVisitor visitor;
    visitor.variables.insert({"x", bit_type.copy()});

    EXPECT_NO_THROW({
        ast.walk(visitor);
    });

    FuncType expected_type(
        std::make_unique<QubitType>(mock_dimvar_expr(4)),
        std::make_unique<QubitType>(mock_dimvar_expr(4)),
        /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}

// '1101'.prep
TEST(qpuTypeCheckingPrepare, prepQlit) {
    Prepare ast(mock_dbg(1),
        std::make_unique<BiTensor>(mock_dbg(2),
            std::make_unique<QubitLiteral>(mock_dbg(3),
                MINUS, Z, mock_dimvar_expr(2)),
            std::make_unique<BiTensor>(mock_dbg(4),
                std::make_unique<QubitLiteral>(mock_dbg(5),
                    PLUS, Z, mock_dimvar_expr(1)),
                std::make_unique<QubitLiteral>(mock_dbg(6),
                    MINUS, Z, mock_dimvar_expr(1)))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FuncType expected_type(
        std::make_unique<QubitType>(mock_dimvar_expr(4)),
        std::make_unique<QubitType>(mock_dimvar_expr(4)),
        /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}

// {'1101'}.prep
TEST(qpuTypeCheckingPrepare, prepBlit) {
    Prepare ast(mock_dbg(1),
        std::make_unique<BasisLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<BiTensor>(mock_dbg(2),
                    std::make_unique<QubitLiteral>(mock_dbg(3),
                        MINUS, Z, mock_dimvar_expr(2)),
                    std::make_unique<BiTensor>(mock_dbg(4),
                        std::make_unique<QubitLiteral>(mock_dbg(5),
                            PLUS, Z, mock_dimvar_expr(1)),
                        std::make_unique<QubitLiteral>(mock_dbg(6),
                            MINUS, Z, mock_dimvar_expr(1)))))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FuncType expected_type(
        std::make_unique<QubitType>(mock_dimvar_expr(4)),
        std::make_unique<QubitType>(mock_dimvar_expr(4)),
        /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}

// {'1101','1111'}.prep
TEST(qpuTypeCheckingPrepare, prepBlitNonSingleton) {
    Prepare ast(mock_dbg(1),
        std::make_unique<BasisLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<BiTensor>(mock_dbg(2),
                    std::make_unique<QubitLiteral>(mock_dbg(3),
                        MINUS, Z, mock_dimvar_expr(2)),
                    std::make_unique<BiTensor>(mock_dbg(4),
                        std::make_unique<QubitLiteral>(mock_dbg(5),
                            PLUS, Z, mock_dimvar_expr(1)),
                        std::make_unique<QubitLiteral>(mock_dbg(6),
                            MINUS, Z, mock_dimvar_expr(1)))),
                std::make_unique<QubitLiteral>(mock_dbg(7),
                    MINUS, Z, mock_dimvar_expr(4)))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "A basis passed to .prep must be a singleton basis");
}

////////////////////////// LIFT BITS //////////////////////////

// Given x = bit[4](0b1101),
// x.q
TEST(qpuTypeCheckingLiftBits, valid) {
    LiftBits ast(mock_dbg(1),
        std::make_unique<Variable>(mock_dbg(2), "x"));

    BitType bit_type(mock_dimvar_expr(4));
    QpuTypeCheckVisitor visitor;
    visitor.variables.insert({"x", bit_type.copy()});

    EXPECT_NO_THROW({
        ast.walk(visitor);
    });

    QubitType expected_type(mock_dimvar_expr(4));
    EXPECT_EQ(*ast.type, expected_type);
}

// '1101'.q
TEST(qpuTypeCheckingLiftBits, invalid) {
    LiftBits ast(mock_dbg(1),
        std::make_unique<BiTensor>(mock_dbg(2),
            std::make_unique<QubitLiteral>(mock_dbg(3),
                MINUS, Z, mock_dimvar_expr(2)),
            std::make_unique<BiTensor>(mock_dbg(4),
                std::make_unique<QubitLiteral>(mock_dbg(5),
                    PLUS, Z, mock_dimvar_expr(1)),
                std::make_unique<QubitLiteral>(mock_dbg(6),
                    MINUS, Z, mock_dimvar_expr(1)))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "The operand x of a bit lift x.q must have type bit[N], not Qubit[4]");
}

////////////////////////// PIPES //////////////////////////

// '0'[4] | id[4]
TEST(qpuTypeCheckingPipe, inputMatch) {
    Pipe ast(mock_dbg(1),
        std::make_unique<QubitLiteral>(mock_dbg(2), PLUS, Z,
                                       mock_dimvar_expr(4)),
        std::make_unique<Identity>(mock_dbg(3), mock_dimvar_expr(4)));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    QubitType expected_type(mock_dimvar_expr(4));
    EXPECT_EQ(*ast.type, expected_type);
}

// '0'[4] | id[3]
TEST(qpuTypeCheckingPipe, inputMismatch) {
    Pipe ast(mock_dbg(1),
        std::make_unique<QubitLiteral>(mock_dbg(2), PLUS, Z,
                                       mock_dimvar_expr(4)),
        std::make_unique<Identity>(mock_dbg(3), mock_dimvar_expr(3)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Input to pipe Qubit[4] does not match right-hand input type Qubit[3]");
}

// '0'[4] | '-'[4]
TEST(qpuTypeCheckingPipe, rhsNotFunc) {
    Pipe ast(mock_dbg(1),
        std::make_unique<QubitLiteral>(mock_dbg(2), PLUS, Z,
                                       mock_dimvar_expr(4)),
        std::make_unique<QubitLiteral>(mock_dbg(3), MINUS, X,
                                       mock_dimvar_expr(4)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Right-hand side of pipe must be a function but got Qubit[4]");
}

////////////////////////// BASIS TRANSLATIONS //////////////////////////

// {'+'[7]} >> {-'+'[7]}
// aka
// {'+'[7]} >> {'+'[7] @ rad(pi)}
TEST(qpuTypeCheckingBtrans, diffuserOk) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<BasisLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<QubitLiteral>(mock_dbg(3),
                    PLUS, X, mock_dimvar_expr(7)))),
        std::make_unique<BasisLiteral>(mock_dbg(4),
            unique_vec<ASTNode>(
                std::make_unique<Phase>(mock_dbg(5),
                    std::make_unique<FloatLiteral>(mock_dbg(7), M_PI),
                    std::make_unique<QubitLiteral>(mock_dbg(8),
                        PLUS, X, mock_dimvar_expr(7))))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FuncType expected_type(std::make_unique<QubitType>(mock_dimvar_expr(7)),
                           std::make_unique<QubitType>(mock_dimvar_expr(7)),
                           /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}

// '+'[7] >> -'+'[7]
TEST(qpuTypeCheckingBtrans, diffuserWrapBasisOk) {
    BasisTranslation ast(mock_dbg(1),
            std::make_unique<QubitLiteral>(mock_dbg(2),
                PLUS, X, mock_dimvar_expr(7)),
            std::make_unique<Phase>(mock_dbg(3),
                std::make_unique<FloatLiteral>(mock_dbg(5), M_PI),
                std::make_unique<QubitLiteral>(mock_dbg(6),
                    PLUS, X, mock_dimvar_expr(7))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FuncType expected_type(std::make_unique<QubitType>(mock_dimvar_expr(7)),
                           std::make_unique<QubitType>(mock_dimvar_expr(7)),
                           /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}


// '+'[6] >> -'+'[7]
// aka
// {'+'[6]} >> {'+'[7] @ rad(pi)}
TEST(qpuTypeCheckingBtrans, diffuserDimMismatch) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<BasisLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<QubitLiteral>(mock_dbg(3),
                    PLUS, X, mock_dimvar_expr(6)))),
        std::make_unique<BasisLiteral>(mock_dbg(4),
            unique_vec<ASTNode>(
                std::make_unique<Phase>(mock_dbg(5),
                    std::make_unique<FloatLiteral>(mock_dbg(7), M_PI),
                    std::make_unique<QubitLiteral>(mock_dbg(8),
                        PLUS, X, mock_dimvar_expr(7))))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Input and output bases of basis translation must have the same number of qubits, but 6 != 7");
}

// '+'[7] >> -'0'[7]
// aka
// {'+'[7]} >> {'0'[7] @ rad(pi)}
TEST(qpuTypeCheckingBtrans, diffuserSpanMismatch) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<BasisLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<QubitLiteral>(mock_dbg(3),
                    PLUS, X, mock_dimvar_expr(7)))),
        std::make_unique<BasisLiteral>(mock_dbg(4),
            unique_vec<ASTNode>(
                std::make_unique<Phase>(mock_dbg(5),
                    std::make_unique<FloatLiteral>(mock_dbg(7), M_PI),
                    std::make_unique<QubitLiteral>(mock_dbg(8),
                        PLUS, Z, mock_dimvar_expr(7))))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Input and output bases of basis translation do not span the same subspace of qubits");
}

// '0' >> pm
// aka
// {'0'} >> pm
TEST(qpuTypeCheckingBtrans, lonely0ToPm) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<BasisLiteral>(mock_dbg(2),
            unique_vec<ASTNode>(
                std::make_unique<QubitLiteral>(mock_dbg(3),
                    PLUS, Z, mock_dimvar_expr(1)))),
        std::make_unique<BuiltinBasis>(mock_dbg(4),
            X, mock_dimvar_expr(1)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Input and output bases of basis translation do not span the same subspace of qubits");
}

// id >> std
TEST(qpuTypeCheckingBtrans, operandNotBasis) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<Identity>(mock_dbg(2),
            mock_dimvar_expr(1)),
        std::make_unique<BuiltinBasis>(mock_dbg(4),
            Z, mock_dimvar_expr(1)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Expected basis but got (Qubit—ʳᵉᵛ→Qubit)");
}

// () >> std
TEST(qpuTypeCheckingBtrans, unitOperand) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<TupleLiteral>(mock_dbg(2)),
        std::make_unique<BuiltinBasis>(mock_dbg(3),
            Z, mock_dimvar_expr(1)));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Expected basis but got Unit");
}

// () + std + () + () >> () + (pm + ())
// specifically, in tree form:
//         >>
//       /   \
//      +     +
//     / \   / \
//    +  () ()  +
//   / \       / \
//  () std    pm  ()
TEST(qpuTypeCheckingBtrans, unitBasisBiTensor) {
    BasisTranslation ast(mock_dbg(1),
        std::make_unique<BiTensor>(mock_dbg(2),
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<TupleLiteral>(mock_dbg(2)),
                std::make_unique<BuiltinBasis>(mock_dbg(2),
                    Z, mock_dimvar_expr(1))),
            std::make_unique<TupleLiteral>(mock_dbg(2))),
        std::make_unique<BiTensor>(mock_dbg(2),
            std::make_unique<TupleLiteral>(mock_dbg(2)),
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<BuiltinBasis>(mock_dbg(2),
                    Z, mock_dimvar_expr(1)),
                std::make_unique<TupleLiteral>(mock_dbg(2)))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    FuncType expected_type(std::make_unique<QubitType>(mock_dimvar_expr(1)),
                           std::make_unique<QubitType>(mock_dimvar_expr(1)),
                           /*is_rev=*/true);
    EXPECT_EQ(*ast.type, expected_type);
}

////////////////////////// BASIS LITERALS //////////////////////////

// {}
TEST(qpuTypeCheckingBlit, empty) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>());

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Basis literal cannot be empty");
}

// {'0','j'}
TEST(qpuTypeCheckingBlit, basisMismatch) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<QubitLiteral>(mock_dbg(2),
                PLUS, Z, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(3),
                MINUS, Y, mock_dimvar_expr(1))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "States in a basis literal must consist of vectors only from std[N], or from only ij[N] instead, or from only pm[N] instead. Yet the first vector is from std[N] and index 1 is from ij[N]");
}

// {'0','11'}
TEST(qpuTypeCheckingBlit, dimMismatch) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<QubitLiteral>(mock_dbg(3),
                PLUS, Z, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(4),
                MINUS, Z, mock_dimvar_expr(2))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "All qubit literals in a basis literal must have the same number of qubits, yet first state has dimension 1 but index 1 has dimension 2");
}

// {'0', '0'|std.flip}
TEST(qpuTypeCheckingBlit, nonLiteral) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<QubitLiteral>(mock_dbg(2),
                PLUS, Z, mock_dimvar_expr(1)),
            std::make_unique<Pipe>(mock_dbg(3),
                std::make_unique<QubitLiteral>(mock_dbg(4),
                    PLUS, Z, mock_dimvar_expr(1)),
                std::make_unique<Flip>(mock_dbg(5),
                    std::make_unique<BuiltinBasis>(mock_dbg(5),
                        Z, mock_dimvar_expr(1))))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Elements of basis literals must be qubit literals, but the element at index 1 is not");
}

// {id}
TEST(qpuTypeCheckingBlit, notQubitTyped) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<Identity>(mock_dbg(2), mock_dimvar_expr(1))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Elements of basis literals must be qubit literals, but the element at index 0 is not");
}

// {()}
TEST(qpuTypeCheckingBlit, unit) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<TupleLiteral>(mock_dbg(3))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Elements of basis literals must have type Qubit[N], but the element at index 0 has type Unit instead");
}

// {''}
TEST(qpuTypeCheckingBlit, emptyQlit) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<QubitLiteral>(mock_dbg(3),
                PLUS, Z, mock_dimvar_expr(0))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Elements of basis literals cannot be empty yet the element at index 0 is");
}

// {'0', '0'}
TEST(qpuTypeCheckingBlit, duplicate) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<QubitLiteral>(mock_dbg(2),
                PLUS, Z, mock_dimvar_expr(1)),
            std::make_unique<QubitLiteral>(mock_dbg(3),
                PLUS, Z, mock_dimvar_expr(1))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Elements of basis literals must be distinct (excluding phases), but the element at index 1 is a repeat");
}

// {'0', -'0'}
TEST(qpuTypeCheckingBlit, duplicateWithPhase) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<QubitLiteral>(mock_dbg(2),
                PLUS, Z, mock_dimvar_expr(1)),
            std::make_unique<Phase>(mock_dbg(3),
                std::make_unique<FloatLiteral>(mock_dbg(5), M_PI),
                std::make_unique<QubitLiteral>(mock_dbg(6),
                    PLUS, Z, mock_dimvar_expr(1)))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Elements of basis literals must be distinct (excluding phases), but the element at index 1 is a repeat");
}

// {'01'}
TEST(qpuTypeCheckingBlit, vec01) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<QubitLiteral>(mock_dbg(3),
                    PLUS, Z, mock_dimvar_expr(1)),
                std::make_unique<QubitLiteral>(mock_dbg(4),
                    MINUS, Z, mock_dimvar_expr(1)))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    BasisType expected_type(mock_dimvar_expr(2));
    ASSERT_EQ(*ast.type, expected_type);

    SpanList expected_span;
    expected_span.append(
        std::make_unique<VeclistSpan>(Z,
            std::initializer_list<llvm::APInt>{
                llvm::APInt(/*numBits=*/2, /*val=*/0b01, /*isSigned=*/false), // '01'
            }));
    BasisType &actual_type = static_cast<BasisType &>(*ast.type);
    EXPECT_EQ(actual_type.span, expected_span);
}

// {'01'[4]}
TEST(qpuTypeCheckingBlit, vec01Broadcast4) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<BroadcastTensor>(mock_dbg(2),
                std::make_unique<BiTensor>(mock_dbg(3),
                    std::make_unique<QubitLiteral>(mock_dbg(4),
                        PLUS, Z, mock_dimvar_expr(1)),
                    std::make_unique<QubitLiteral>(mock_dbg(5),
                        MINUS, Z, mock_dimvar_expr(1))),
                mock_dimvar_expr(4))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    BasisType expected_type(mock_dimvar_expr(8));
    ASSERT_EQ(*ast.type, expected_type);

    SpanList expected_span;
    expected_span.append(
        std::make_unique<VeclistSpan>(Z,
            std::initializer_list<llvm::APInt>{
                llvm::APInt(/*numBits=*/8, /*val=*/0b01010101, /*isSigned=*/false), // '01010101'
            }));
    BasisType &actual_type = static_cast<BasisType &>(*ast.type);
    EXPECT_EQ(actual_type.span, expected_span);
}

// {'0-'}
TEST(qpuTypeCheckingBlit, vec0m) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<QubitLiteral>(mock_dbg(2),
                    PLUS, Z, mock_dimvar_expr(1)),
                std::make_unique<QubitLiteral>(mock_dbg(3),
                    MINUS, X, mock_dimvar_expr(1)))));

    QW_EXPECT_THROW_MSG({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    }, TypeException,
    "Qubit literals in a basis literal must not mix vectors of std[N], ij[N], and pm[N], but a vector from std[N] and a vector from pm[N] are mixed in the vector at index 0");
}

// {'1'+()}
TEST(qpuTypeCheckingBlit, vec1unit) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<QubitLiteral>(mock_dbg(2),
                    MINUS, Z, mock_dimvar_expr(1)),
                std::make_unique<TupleLiteral>(mock_dbg(2)))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    BasisType expected_type(mock_dimvar_expr(1));
    ASSERT_EQ(*ast.type, expected_type);

    SpanList expected_span;
    expected_span.append(
        std::make_unique<VeclistSpan>(Z,
            std::initializer_list<llvm::APInt>{
                llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '01'
            }));
    BasisType &actual_type = static_cast<BasisType &>(*ast.type);
    EXPECT_EQ(actual_type.span, expected_span);
}

// {()+'1'}
TEST(qpuTypeCheckingBlit, vecunit1) {
    BasisLiteral ast(mock_dbg(1),
        unique_vec<ASTNode>(
            std::make_unique<BiTensor>(mock_dbg(2),
                std::make_unique<TupleLiteral>(mock_dbg(2)),
                std::make_unique<QubitLiteral>(mock_dbg(2),
                    MINUS, Z, mock_dimvar_expr(1)))));

    EXPECT_NO_THROW({
        QpuTypeCheckVisitor visitor;
        ast.walk(visitor);
    });

    BasisType expected_type(mock_dimvar_expr(1));
    ASSERT_EQ(*ast.type, expected_type);

    SpanList expected_span;
    expected_span.append(
        std::make_unique<VeclistSpan>(Z,
            std::initializer_list<llvm::APInt>{
                llvm::APInt(/*numBits=*/1, /*val=*/0b1, /*isSigned=*/false), // '01'
            }));
    BasisType &actual_type = static_cast<BasisType &>(*ast.type);
    EXPECT_EQ(actual_type.span, expected_span);
}
