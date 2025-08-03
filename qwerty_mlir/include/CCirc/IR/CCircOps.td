//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD

include "CCirc/IR/CCircDialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def CCirc_CircuitOp : CCirc_Op<"circuit", [IsolatedFromAbove]> {
    let summary = "Classical circuit";
    let description = [{
        A classical circuit from m to n bits.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        mlir::Block &bodyBlock() {
            return getBody().front();
        };
    }];
}

def CCirc_ReturnOp : CCirc_Op<"return", [Pure, HasParent<"CircuitOp">,
                                         ReturnLike, Terminator]> {
    let summary = "Terminator for ccirc.circuit";
    let arguments = (ins Variadic<CCirc_WireBundle>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` qualified(type($operands)))?";
}

def CCirc_AndOp : CCirc_Op<"and", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise AND operation";
    let description = [{
        Bitwise AND between two wire bundles.
    }];
    let arguments = (ins CCirc_WireBundle:$left, CCirc_WireBundle:$right);
    let results = (outs CCirc_WireBundle:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_OrOp : CCirc_Op<"or", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise OR operation";
    let description = [{
        Bitwise OR between two wire bundles.
    }];
    let arguments = (ins CCirc_WireBundle:$left, CCirc_WireBundle:$right);
    let results = (outs CCirc_WireBundle:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_XorOp : CCirc_Op<"xor", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise XOR operation";
    let description = [{
        Bitwise XOR between two wire bundles.
    }];
    let arguments = (ins CCirc_WireBundle:$left, CCirc_WireBundle:$right);
    let results = (outs CCirc_WireBundle:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_NotOp : CCirc_Op<"not", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise NOT operation";
    let description = [{
        Bitwise NOT on a single wire bundles.
    }];
    let arguments = (ins CCirc_WireBundle:$operand);
    let results = (outs CCirc_WireBundle:$result);
    let assemblyFormat = "`(` $operand `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
