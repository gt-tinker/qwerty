//===- QwertyInterfaces.td - Qwerty dialect interfaces/traits ----*- tablegen -*-===//
//===------------------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_INTERFACES_TD
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_INTERFACES_TD

include "mlir/IR/Interfaces.td"

def Predicatable : OpInterface<"PredicatableOpInterface"> {
    let description = [{
        Operations that can be predicated on a basis implement this interface
    }];
    let cppNamespace = "::qwerty";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Determines whether this op can be predicated on a basis. This
                is useful when an op may be predicatable or not depending on
                the context.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isPredicatable",
            /*args=*/(ins),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return true; }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Builds a controlled version of this op if isPredicatable() is
                true. The predicating qbundle is passed through predIn, and the
                new predicating qbundle must be returned through predOut.
                The new outputs must be written to newOutputs (predicating
                qubits excluded). As a special case, these new outputs may be
                exactly the existing results, in which case the IR will not be
                changed.

                The default implementation leaves the instruction as-is. This
                is useful for ops that are not Stationary yet do not need
                explicit predication.
            }],
            /*retTy=*/"void",
            /*methodName=*/"buildPredicated",
            /*args=*/(ins "::mlir::RewriterBase&":$rewriter,
                          "::qwerty::BasisAttr":$predBasis,
                          "::mlir::Value":$predIn,
                          "::mlir::Value&":$predOut,
                          "::mlir::ValueRange":$newInputs,
                          "::llvm::SmallVectorImpl<::mlir::Value>&":$newOutputs),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
                newOutputs.clear();
                newOutputs.append($_op->result_begin(), $_op->result_end());
            }]
        >
    ];

    let verify = [{
        // It's a shame this can't be a static_assert, but we do a runtime
        // assert here because apparently trait registration is compile-time
        // but interface registration is at runtime.
        assert(llvm::isa<qcirc::AdjointableOpInterface>($_op)
               && "A predicatable op must be adjointable");
        return mlir::success();
    }];
}

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_INTERFACES_TD
