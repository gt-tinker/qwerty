//===- QwertyOps.td - Qwerty dialect types -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_TYPES_TD
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_TYPES_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "QCirc/IR/QCircInterfaces.td"
include "Qwerty/IR/QwertyDialect.td"

// Base quantum type definition.
class Qwerty_Type<string name, string typeMnemonic, list<Trait> traits = []> :
        TypeDef<Qwerty_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Qwerty_BitBundle : Qwerty_Type<"BitBundle", "bitbundle"> {
    let summary = "Fixed-size tuple of bits";
    let description = [{
        Constant-sized list of bits
    }];
    let parameters = (ins "uint64_t":$dim);
    let assemblyFormat = "`[` $dim `]`";
    let genVerifyDecl = 1;
}

def Qwerty_QBundle : Qwerty_Type<"QBundle", "qbundle", [
        DeclareTypeInterfaceMethods<NonStationary, ["getNumQubits"]>]> {
    let summary = "Fixed-size tuple of qubits";
    let description = [{
        Constant-sized list of qubits
    }];
    let parameters = (ins "uint64_t":$dim);
    let assemblyFormat = "`[` $dim `]`";
    let genVerifyDecl = 1;
}

def Qwerty_FunctionType : Qwerty_Type<"Function", "func"> {
    let summary = "Function with quantum metadata";
    let description = [{
        An MLIR function type plus a flag denoting reversibility. This models
        the Qwerty type system, which has both T->U and T-rev->U.
    }];

    let parameters = (ins "mlir::FunctionType":$function_type,
                          "bool":$reversible);

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_TYPES_TD
