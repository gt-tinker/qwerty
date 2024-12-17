#ifndef TEST_SUPPORT_H
#define TEST_SUPPORT_H

#include "ast.hpp"

// Originally I used this:
// https://stackoverflow.com/a/66260986/321301
// But then I hit this: https://github.com/google/googletest/issues/4073
// (i.e., gtest is unexpectedly calling the test code twice)
//
// So I settled on this hackier solution:
// https://stackoverflow.com/a/40561685/321301
#define QW_EXPECT_THROW_MSG(stmt, type, msg) \
    EXPECT_THROW({ \
        try { \
            stmt \
        } catch (const type &exc) { \
            EXPECT_STREQ(exc.what(), msg); \
            throw; \
        } \
    }, type)

// Hacky replacement for std::initializer_list
// https://stackoverflow.com/a/73674943/321301
template<typename T, typename ...Args>
std::vector<std::unique_ptr<T>> unique_vec(Args... args) {
    std::vector<std::unique_ptr<T>> vec;
    (vec.push_back(std::move(args)), ...);
    return vec;
}

std::unique_ptr<DebugInfo> mock_dbg(unsigned int row);
std::unique_ptr<DimVarExpr> mock_dimvar_expr(DimVarValue dvv);

#endif
