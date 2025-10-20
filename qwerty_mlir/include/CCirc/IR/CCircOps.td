//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD

include "CCirc/IR/CCircDialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def CCirc_CircuitOp : CCirc_Op<"circuit", [IsolatedFromAbove, Symbol]> {
    let summary = "Classical circuit";
    let description = [{
        A classical circuit from m to n bits.
    }];

    let arguments = (ins BoolAttr:$reversible,
                         SymbolNameAttr:$sym_name,
                         OptionalAttr<StrAttr>:$sym_visibility);

    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "bool":$reversible, "::llvm::StringRef":$sym_name), [{
            build($_builder, $_state, reversible, sym_name, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::Block &bodyBlock() {
            return getBody().front();
        }
        uint64_t inDim();
        uint64_t outDim();
        CircuitOp buildInverseCircuit(mlir::RewriterBase &rewriter,
                                      mlir::Location loc,
                                      llvm::StringRef inv_circ_name);
    }];
}

def CCirc_ReturnOp : CCirc_Op<"return", [Pure, HasParent<"CircuitOp">,
                                         ReturnLike, Terminator]> {
    let summary = "Terminator for ccirc.circuit";
    let arguments = (ins Variadic<CCirc_Wire>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` qualified(type($operands)))?";
}

def CCirc_ConstantOp : CCirc_Op<"constant", [Pure, ConstantLike, InferTypeOpAdaptor]> {
    let summary = "Constant bits";
    let description = [{
        Arbitrary-length constant wires
    }];

    let builders = [
        OpBuilder<(ins "::llvm::APInt":$val), [{
            build($_builder, $_state, $_builder.getIntegerAttr($_builder.getIntegerType(val.getBitWidth()), val));
        }]>
    ];

    let arguments = (ins APIntAttr:$value);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
    let hasVerifier = 1;
    let hasFolder = 1;
}

def CCirc_AndOp : CCirc_Op<"and", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise AND operation";
    let description = [{
        Bitwise AND between two wire bundles.
    }];
    let arguments = (ins CCirc_Wire:$left, CCirc_Wire:$right);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def CCirc_OrOp : CCirc_Op<"or", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise OR operation";
    let description = [{
        Bitwise OR between two wire bundles.
    }];
    let arguments = (ins CCirc_Wire:$left, CCirc_Wire:$right);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_XorOp : CCirc_Op<"xor", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise XOR operation";
    let description = [{
        Bitwise XOR between two wire bundles.
    }];
    let arguments = (ins CCirc_Wire:$left, CCirc_Wire:$right);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $left `,` $right `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_NotOp : CCirc_Op<"not", [Pure, InferTypeOpAdaptor]> {
    let summary = "Bitwise NOT operation";
    let description = [{
        Bitwise NOT on a single wire bundles.
    }];
    let arguments = (ins CCirc_Wire:$operand);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $operand `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def CCirc_ParityOp : CCirc_Op<"parity", [Pure, InferTypeOpAdaptor]> {
    let summary = "Parity operation";
    let description = [{
        Compute the bitwise XOR of all operands. Useful for making a high-level XAG.
    }];
    let arguments = (ins Variadic<CCirc_Wire>:$operands);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $operands `)` attr-dict `:` functional-type(operands, results)";
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

def CCirc_RotateLeftOp : CCirc_Op<"rotl", [Pure, InferTypeOpAdaptor]> {
    let summary = "Perform rotl(value, amount)";
    let arguments = (ins CCirc_Wire:$value, CCirc_Wire:$amount);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $value `,` $amount `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_RotateRightOp : CCirc_Op<"rotr", [Pure, InferTypeOpAdaptor]> {
    let summary = "Perform rotr(value, amount)";
    let arguments = (ins CCirc_Wire:$value, CCirc_Wire:$amount);
    let results = (outs CCirc_Wire:$result);
    let assemblyFormat = "`(` $value `,` $amount `)` attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def CCirc_ModMulOp : CCirc_Op<"modmul", [Pure, InferTypeOpAdaptor]> {
    let summary = "Modular multiplier";
    let description = [{
        Modular multiplication in-place
    }];
    let arguments = (ins I64Attr:$x, I64Attr:$j, I64Attr:$modN, CCirc_Wire:$y);
    let results = (outs CCirc_Wire:$product);
    let assemblyFormat = "$x $j $modN $y attr-dict `:` functional-type(operands, results)";
    let hasVerifier = 1;
}

def Qwerty_WirePackOp : CCirc_Op<"wirepack", [
        Pure, InferTypeOpAdaptor]> {
    let summary = "Combine wire bundles into a wire[N]";
    let arguments = (ins Variadic<CCirc_Wire>:$wires);
    let results = (outs CCirc_Wire:$wire);
    let hasCanonicalizer = 1;
    let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_WireUnpackOp : CCirc_Op<"wireunpack", [
        Pure, InferTypeOpAdaptor]> {
    let summary = "Split a wire[N] into N wire[1]s";
    let arguments = (ins CCirc_Wire:$wire);
    let results = (outs Variadic<CCirc_Wire>:$wires);
    let hasCanonicalizer = 1;
    let assemblyFormat = "$wire attr-dict `:` functional-type(operands, results)";
}

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_OPS_TD
