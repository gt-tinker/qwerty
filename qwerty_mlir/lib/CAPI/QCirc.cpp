#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "CAPI/QCirc.h"
#include "QCirc/IR/QCircDialect.h"
#include "QCirc/Utils/QCircUtils.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QCirc, qcirc, qcirc::QCircDialect)

MlirLogicalResult mlirQCircGenerateQasm(
        MlirOperation func_op, bool print_locs, char **result_out) {
    *result_out = nullptr;

    mlir::Operation *func_op_unwrap = unwrap(func_op);
    mlir::func::FuncOp func_op_op =
        llvm::cast<mlir::func::FuncOp>(func_op_unwrap);

    std::string result;
    mlir::LogicalResult ret = qcirc::generateQasm(func_op_op, print_locs, result);

    char *result_buf = new char[result.length() + 1];
    std::strcpy(result_buf, result.c_str());
    *result_out = result_buf;

    return wrap(ret);
}

void mlirQCircGenerateQasmDestroyBuf(char *result_buf) {
    if (result_buf) {
        delete result_buf;
    }
}
