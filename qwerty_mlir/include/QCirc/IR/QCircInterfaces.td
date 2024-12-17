//===- QCircInterfaces.td - QCirc dialect interfaces/traits ----*- tablegen -*-===//
//===------------------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_INTERFACES_TD
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_INTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"

// A Value with this type is both taken in and returned by a nonstationary
// operation(*), such as a quantum gate. If the Value is the type of an
// operand, there must be a result with the same type, and vice versa(*). It is
// usually a qubit.
// (*) Unless this operation is a source or sink.
def NonStationary : TypeInterface<"NonStationaryTypeInterface"> {
    let cppNamespace = "::qcirc";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Returns the number of qubits this type holds. The default
                behavior is to return 1.
            }],
            /*retTy=*/"uint64_t",
            /*methodName=*/"getNumQubits",
            /*args=*/(ins),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return 1; }]
        >
    ];
}

// Does not directly affect qubits. Specifically, when taking the adjoint of
// something, you can ignore this op entirely
def Stationary : NativeOpTrait<"IsStationaryOpTrait"> {
    let cppNamespace = "::qcirc";
}

def Adjointable : OpInterface<"AdjointableOpInterface"> {
    let description = [{
        Operations that are adjointable have this interface, used for them to
        spit out the adjoint version of themselves
    }];
    let cppNamespace = "::qcirc";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Determines whether this op is adjointable. This is useful when
                an op may be reversible or not depending on the context.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isAdjointable",
            /*args=*/(ins),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return true; }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns number of operands this op will have once adjointed.
                Returning zero means this is a sink node that will become a
                source node. The default is to return the number of (forward)
                operands. Note that deviating from this default means that
                stationary operands are not permitted.
            }],
            /*retTy=*/"unsigned",
            /*methodName=*/"getNumOperandsOfAdjoint",
            /*args=*/(ins),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
                return $_op->getNumOperands();
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns true if the operand index provided is a stationary
                operand. Default is to check whether the type of the operand in
                question does not have the NonStationaryTypeInterface (unless
                getNumOperandsOfAdjoint() does not match the number of
                operands, in which case the default behavior would be unsafe.)
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isStationaryOperand",
            /*args=*/(ins "unsigned":$operandIndex),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
                return getNumOperandsOfAdjoint() == $_op->getNumOperands()
                       && !llvm::isa<qcirc::NonStationaryTypeInterface>(
                              $_op->getOperand(operandIndex).getType());
            }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Builds the reverse of this op if isAdjointable() is true. The
                new inputs (formerly outputs) are passed through newInputs. The
                new outputs (formerly inputs) are written to newOutputs.
            }],
            /*retTy=*/"void",
            /*methodName=*/"buildAdjoint",
            /*args=*/(ins "::mlir::RewriterBase&":$rewriter,
                          "::mlir::ValueRange":$newInputs,
                          "::llvm::SmallVectorImpl<::mlir::Value>&":$newOutputs)
        >
    ];
}

def Controllable : OpInterface<"ControllableOpInterface"> {
    let description = [{
        Operations that can be controlled implement this interface
    }];
    let cppNamespace = "::qcirc";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Determines whether this op is controllable. This is useful when
                an op may be controllable or not depending on the context.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isControllable",
            /*args=*/(ins),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{ return true; }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Builds a controlled version of this op if isControllable() is
                true. The control qubits are passed through controlsIn, and the
                resulting control qubits must be returned through controlsOut.
                The new outputs must be written to newOutputs.
            }],
            /*retTy=*/"void",
            /*methodName=*/"buildControlled",
            /*args=*/(ins "::mlir::RewriterBase&":$rewriter,
                          "::mlir::ValueRange":$controlsIn,
                          "::llvm::SmallVectorImpl<::mlir::Value>&":$controlsOut,
                          "::mlir::ValueRange":$newInputs,
                          "::llvm::SmallVectorImpl<::mlir::Value>&":$newOutputs)
        >
    ];

    let verify = [{
        // It's a shame this can't be a static_assert, but we do so here
        // because apparently trait registration is compile-time but interface
        // registration is at runtime.
        assert(llvm::isa<qcirc::AdjointableOpInterface>($_op)
               && "A controllable op must be adjointable");
        return mlir::success();
    }];
}

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_INTERFACES_TD
