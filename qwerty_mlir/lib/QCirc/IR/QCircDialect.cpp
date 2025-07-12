//===- QCircDialect.cpp - QCirc dialect ---------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"

#include "QCirc/IR/QCircDialect.h"
#include "QCirc/IR/QCircAttributes.h"
#include "QCirc/IR/QCircOps.h"

#include "QCirc/IR/QCircOpsDialect.cpp.inc"
#include "QCirc/IR/QCircOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// QCirc dialect.
//===----------------------------------------------------------------------===//

namespace {
struct QCircInlinerInterface : public mlir::DialectInlinerInterface {
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    /// All QCirc dialect ops can be inlined.
    bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::IRMapping &) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                         bool wouldBeCloned, mlir::IRMapping &valueMapping) const final {
        return true;
    }
};

// Aggravating trick to create a llvm::ConstantArray in MLIR. See the
// comment in QCircOps.td for LLVMConstantArrayOp.
class QCircLLVMIRTranslationInterface
    : public mlir::LLVMTranslationDialectInterface {
public:
    using mlir::LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

    mlir::LogicalResult convertOperation(
            mlir::Operation *op,
            llvm::IRBuilderBase &builder,
            mlir::LLVM::ModuleTranslation &mod_trans) const final {
        if (qcirc::LLVMConstantArrayOp const_arr =
                llvm::dyn_cast<qcirc::LLVMConstantArrayOp>(op)) {

            llvm::SmallVector<llvm::Constant *> elems;
            for (mlir::Value operand : const_arr.getElements()) {
                elems.push_back(llvm::cast<llvm::Constant>(
                    mod_trans.lookupValue(operand)));
            }

            llvm::Type *llvm_ty = mod_trans.convertType(
                const_arr.getResult().getType());
            llvm::ArrayType *arr_ty = llvm::cast<llvm::ArrayType>(llvm_ty);

            llvm::Constant *llvm_const =
                llvm::ConstantArray::get(arr_ty, elems);
            mod_trans.mapValue(const_arr.getResult(), llvm_const);
            return mlir::success();
        }

        return mlir::failure();
    }
};

} // namespace

namespace qcirc {

void QCircDialect::initialize() {
    registerAttributes();
    registerTypes();

    addOperations<
#define GET_OP_LIST
#include "QCirc/IR/QCircOps.cpp.inc"
    >();

    addInterfaces<QCircInlinerInterface,
                  QCircLLVMIRTranslationInterface>();
}

void registerQCircDialectTranslation(mlir::DialectRegistry &registry) {
    // This will actually register the translation interface in
    // QCircDialect::initialize() (see above)
    registry.insert<QCircDialect>();
    // ...so we don't need to do something like this, even though the `llvm'
    // dialect in MLIR does (why? the LLVM dialect does not register this
    // interface itself as we do above):
    //registry.addExtension(+[](mlir::MLIRContext *ctx, QCircDialect *dialect) {
    //    dialect->addInterfaces<QCircLLVMIRTranslationInterface>();
    //});
}

} // namespace qcirc
