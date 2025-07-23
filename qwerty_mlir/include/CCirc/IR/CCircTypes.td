//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD

include "CCirc/IR/CCircDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// class OperationType

//===----------------------------------------------------------------------===//
// Base ccirc type definition.
//===----------------------------------------------------------------------===//

class CCirc_Type<string name, string typeMnemonic, list<Trait> traits = []> :
        TypeDef<CCirc_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def CCirc_Wire : CCirc_Type<"Wire", "wire"> {
    let summary = "Wire";
    let description = [{
        Opaque type for a single two-level quantum state
    }];
}

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD
