//===- QCircOps.h - QCirc dialect ops -----------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_OPS_H
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/CommonFolders.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircTypes.h"
#include "QCirc/IR/QCircAttributes.h"
#include "QCirc/IR/QCircInterfaces.h"

#define GET_OP_CLASSES
#include "QCirc/IR/QCircOps.h.inc"

namespace qcirc {

// Very common. Wraps an arith.const inside a qcirc.calc op.
mlir::Value stationaryF64Const(mlir::OpBuilder &builder, mlir::Location loc, double theta);

// Very common in implementations of rebuildAdjoint() in both the QCirc and
// Qwerty dialects. Wraps an arith.negf inside a qcirc.calc op.
mlir::Value stationaryF64Negate(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value theta);

// Wrap some ops created by a callback in a qcirc.calc op with stationary
// operands.
mlir::Value wrapStationaryF64Ops(
    mlir::OpBuilder &builder,
    mlir::Location loc,
    mlir::ValueRange args,
    std::function<mlir::Value(mlir::ValueRange)> build_body);

// Heavily, heavily inspired by the implementation of mlir::m_Constant() in the
// MLIR source tree. The difference here is that the ConstantLike trait
// requires having (always) having no operands. That is not true of
// qcirc.calc, but there are many cases where we may want to fold it anyway
// (when it _does_ have no arguments).
template <typename AttrT>
struct CalcConstantBinder {
    AttrT *bind_value;

    CalcConstantBinder(AttrT *bind_value) : bind_value(bind_value) {}

    bool match(mlir::Operation *op) {
        auto binder = mlir::m_Constant(bind_value);
        if (binder.match(op)) {
            return true;
        }

        if (qcirc::CalcOp calc = llvm::dyn_cast<qcirc::CalcOp>(op)) {
            if (!calc.getInputs().empty()) {
                return false;
            }

            // This might (MIGHT) be constant-like... try and see
            llvm::SmallVector<mlir::OpFoldResult> folded;
            mlir::LogicalResult result = calc->fold(/*operands=*/{}, folded);
            if (!mlir::succeeded(result)) {
                return false;
            }

            if (folded.size() != 1) {
                return false;
            }

            if (auto attr = llvm::dyn_cast<AttrT>(llvm::cast<mlir::Attribute> (folded.front()))) {
                *bind_value = attr;
                return true;
            }

            return false;
        }

        return false;
    }
};

template <typename AttrT>
CalcConstantBinder<AttrT> m_CalcConstant(AttrT *bind_value) {
    return CalcConstantBinder<AttrT>(bind_value);
}


} // namespace qcirc

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_OPS_H
