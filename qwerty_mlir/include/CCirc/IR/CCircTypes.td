//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD

include "CCirc/IR/CCircDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base ccirc type definition.
//===----------------------------------------------------------------------===//

class CCirc_Type<string name, string typeMnemonic, list<Trait> traits = []> :
        TypeDef<CCirc_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def CCirc_Wire : CCirc_Type<"Wire", "wire"> {
    let summary = "N-bit wire bundle";
    let description = [{
        Constant-sized bundle of Wires
    }];
    let parameters = (ins "uint64_t":$dim);
    let assemblyFormat = "`[` $dim `]`";
    let genVerifyDecl = 1;
}

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD
