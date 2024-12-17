// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include "gtest/gtest.h"
#include "test_support.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"

TEST(flagDynBasis, assignBasis) {
    Assign ast(mock_dbg(1),
               "b",
               std::make_unique<BuiltinBasis>(mock_dbg(2),
                                              X,
                                              mock_dimvar_expr(4)));

    // Need to run typechecking first to get the type metadata set up that
    // FlagDynamicBasisVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        ast.walk(type_check);
    }

    FlagDynamicBasisVisitor visitor;

    QW_EXPECT_THROW_MSG({
        ast.walk(visitor);
    }, TypeException,
    "A basis cannot be used in an expression");
}

TEST(flagDynBasis, varBasis) {
    Variable ast(mock_dbg(1),
                 "b");

    // Need to run typechecking first to get the type metadata set up that
    // FlagDynamicBasisVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        type_check.variables.insert({"b",
            std::make_unique<BasisType>(mock_dimvar_expr(4))});
        ast.walk(type_check);
    }

    FlagDynamicBasisVisitor visitor;

    QW_EXPECT_THROW_MSG({
        ast.walk(visitor);
    }, TypeException,
    "The AST node Variable cannot have type basis");
}

TEST(flagDynBasis, basisRoot) {
    BuiltinBasis ast(mock_dbg(1),
                     X,
                     mock_dimvar_expr(4));

    // Need to run typechecking first to get the type metadata set up that
    // FlagDynamicBasisVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        ast.walk(type_check);
    }

    FlagDynamicBasisVisitor visitor;
    EXPECT_NO_THROW({
        ast.walk(visitor);
    });
}

TEST(flagDynBasis, flipBasis) {
    Flip ast(mock_dbg(1),
             std::make_unique<BuiltinBasis>(mock_dbg(2),
                                            Y,
                                            mock_dimvar_expr(1)));

    // Need to run typechecking first to get the type metadata set up that
    // FlagDynamicBasisVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        ast.walk(type_check);
    }

    FlagDynamicBasisVisitor visitor;
    EXPECT_NO_THROW({
        ast.walk(visitor);
    });
}
