// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include "gtest/gtest.h"
#include "test_support.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"

TEST(flagImmat, assignBasis) {
    Assign ast(mock_dbg(1),
               "b",
               std::make_unique<BuiltinBasis>(mock_dbg(2),
                                              X,
                                              mock_dimvar_expr(4)));

    // Need to run typechecking first to get the type metadata set up that
    // FlagImmaterialVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        ast.walk(type_check);
    }

    FlagImmaterialVisitor visitor;

    QW_EXPECT_THROW_MSG({
        ast.walk(visitor);
    }, TypeException,
    "Type that includes an immaterial types used improperly. Offending "
    "type: Basis[4]");
}

TEST(flagImmat, varBasis) {
    Variable ast(mock_dbg(1),
                 "b");

    // Need to run typechecking first to get the type metadata set up that
    // FlagImmaterialVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        type_check.variables.insert({"b",
            std::make_unique<BasisType>(mock_dimvar_expr(4))});
        ast.walk(type_check);
    }

    FlagImmaterialVisitor visitor;

    QW_EXPECT_THROW_MSG({
        ast.walk(visitor);
    }, TypeException,
    "Type that includes an immaterial types used improperly. Offending "
    "type: Basis[4]");
}

TEST(flagImmat, basisRoot) {
    BuiltinBasis ast(mock_dbg(1),
                     X,
                     mock_dimvar_expr(4));

    // Need to run typechecking first to get the type metadata set up that
    // FlagImmaterialVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        ast.walk(type_check);
    }

    FlagImmaterialVisitor visitor;
    EXPECT_NO_THROW({
        ast.walk(visitor);
    });
}

TEST(flagImmat, flipBasis) {
    Flip ast(mock_dbg(1),
             std::make_unique<BuiltinBasis>(mock_dbg(2),
                                            Y,
                                            mock_dimvar_expr(1)));

    // Need to run typechecking first to get the type metadata set up that
    // FlagImmaterialVisitor expects
    {
        QpuTypeCheckVisitor type_check;
        ast.walk(type_check);
    }

    FlagImmaterialVisitor visitor;
    EXPECT_NO_THROW({
        ast.walk(visitor);
    });
}
