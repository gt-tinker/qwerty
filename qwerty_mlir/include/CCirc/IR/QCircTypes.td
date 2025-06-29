//===- QCircOps.td - QCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_TYPES_TD
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_TYPES_TD

include "QCirc/IR/QCircDialect.td"
include "QCirc/IR/QCircInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// class OperationType

//===----------------------------------------------------------------------===//
// Base qcirc type definition.
//===----------------------------------------------------------------------===//

class QCirc_Type<string name, string typeMnemonic, list<Trait> traits = []> :
        TypeDef<QCirc_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QCirc_Qubit : QCirc_Type<"Qubit", "qubit", [
        DeclareTypeInterfaceMethods<NonStationary>,
        DeclareTypeInterfaceMethods<MemRefElementTypeInterface>]> {
    let summary = "Qubit";
    let description = [{
        Opaque type for a single two-level quantum state
    }];
}

def QCirc_Array : QCirc_Type<"Array", "array"> {
    let summary = "Array of values";
    let description = [{
        Array of elemType. Lowered to a 1D QIR array
    }];
    let parameters = (ins "::mlir::Type":$elemType, "uint64_t":$dim);
    let assemblyFormat = "`<` $elemType `>` `` `[` $dim `]`";
}

def QCirc_Callable : QCirc_Type<"Callable", "callable"> {
    let summary = "Callable value";
    let description = [{
        Pointer to a buffer containing captures and a function pointer. Lowered
        to a a QIR callable
    }];
    let parameters = (ins "::mlir::FunctionType":$functionType);
    let assemblyFormat = "$functionType";
}

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_TYPES_TD
