//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD

include "CCirc/IR/CCircDialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def CCirc_AndOp : CCirc_Op<"and", [
        Pure]> {
    let summary = "A quantum gate with 2 targets and no real-valued parameters";
    let description = [{
        Dataflow semantics two-qubit quantum gate with a variable number of
        control qubits. (That is, qubits flow through this gate like a typical SSA op.)
    }];
    let arguments = (ins CCirc_Wire:$leftwire, CCirc_Wire:$rightwire);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $leftwire `,` $rightwire `)` attr-dict `:` functional-type(operands, results)";
}

def CCirc_OrOp : CCirc_Op<"or", [Pure]> {
    let summary = "Logical OR operation";
    let description = [{
        Bitwise OR between two wires.
    }];
    let arguments = (ins CCirc_Wire:$left, CCirc_Wire:$right);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
}

def CCirc_XorOp : CCirc_Op<"xor", [Pure]> {
    let summary = "Logical XOR operation";
    let description = [{
        Bitwise XOR between two wires.
    }];
    let arguments = (ins CCirc_Wire:$left, CCirc_Wire:$right);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
}

def CCirc_NotOp : CCirc_Op<"not", [Pure]> {
    let summary = "Logical NOT operation";
    let description = [{
        Bitwise NOT on a single wire.
    }];
    let arguments = (ins CCirc_Wire:$operand);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $operand `)` attr-dict `:` functional-type(operands, results)";
    let hasCanonicalizer = 1;
}

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
