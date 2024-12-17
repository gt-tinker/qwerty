#include "test_support.hpp"
#include "ast.hpp"

// We don't link the tester with Python, so stub these out
void DebugInfo::python_incref(void *p) { (void)p; }
void DebugInfo::python_decref(void *p) { (void)p; }

std::unique_ptr<DebugInfo> mock_dbg(unsigned int row) {
    return std::make_unique<DebugInfo>("mock.py", row, 4, nullptr);
}

std::unique_ptr<DimVarExpr> mock_dimvar_expr(DimVarValue dvv) {
    return std::make_unique<DimVarExpr>(dvv);
}
