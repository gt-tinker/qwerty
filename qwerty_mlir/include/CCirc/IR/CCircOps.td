//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD

include "CCirc/IR/CCircDialect.td"
include "CCirc/IR/CCircAttributes.td"

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

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
