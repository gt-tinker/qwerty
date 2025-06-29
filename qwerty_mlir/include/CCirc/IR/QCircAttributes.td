//===- QwertyOps.td - QCirc dialect attributes -----------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_ATTRIBUTES_TD
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_ATTRIBUTES_TD

include "mlir/IR/EnumAttr.td"
include "QCirc/IR/QCircDialect.td"

def QCirc_Gate1QAttr : I64EnumAttr<
        "Gate1Q", "Standard one-qubit gates",
        [
            // Single-qubit gates as defined in Section 4.2 of Nielsen and
            // Chuang. Here, dg = dagger (adjoint)
            I64EnumAttrCase<"X",    0>,
            I64EnumAttrCase<"Y",    1>,
            I64EnumAttrCase<"Z",    2>,
            I64EnumAttrCase<"H",    3>,
            I64EnumAttrCase<"S",    4>,
            I64EnumAttrCase<"Sdg",  5>,
            // Square root of X (present in Tweedledum IR)
            I64EnumAttrCase<"Sx",   6>,
            I64EnumAttrCase<"Sxdg", 7>,
            I64EnumAttrCase<"T",    8>,
            I64EnumAttrCase<"Tdg",  9>,
        ]> {
    let cppNamespace = "::qcirc";
}

def QCirc_Gate1Q1PAttr : I64EnumAttr<
        "Gate1Q1P", "One-qubit gates with 1 parameter",
        [
            // P(θ) = |0⟩⟨0| + e^(iθ)|1⟩⟨1|
            I64EnumAttrCase<"P",    0>,
            // As defined in Section 4.2 of Nielsen and Chuang
            I64EnumAttrCase<"Rx",   1>,
            I64EnumAttrCase<"Ry",   2>,
            I64EnumAttrCase<"Rz",   3>,
        ]> {
    let cppNamespace = "::qcirc";
}

def QCirc_Gate1Q3PAttr : I64EnumAttr<
        "Gate1Q3P", "One-qubit gates with 3 parameters",
        [
            // As defined in OpenQASM 3 by Cross et al. (2022)
            I64EnumAttrCase<"U", 0>,
        ]> {
    let cppNamespace = "::qcirc";
}

def QCirc_Gate2QAttr : I64EnumAttr<
        "Gate2Q", "Standard two-qubit gates",
        [
            // Standard SWAP gate: |00⟩⟨00| + |01⟩⟨10| + |10⟩⟨01| + |11⟩⟨11|
            I64EnumAttrCase<"Swap", 0>,
        ]> {
    let cppNamespace = "::qcirc";
}

def QCirc_FuncSpecAttr : QCirc_Attr<"FuncSpec", "funcspec"> {
    let summary = "A tuple holding a function specialization";
    let description = [{
        Used by CallableMetadataOp to map function specializations to symbol
        names of the implementing functions.
    }];
    let parameters = (ins "bool":$adjoint,
                          "uint64_t":$numControls,
                          "mlir::FlatSymbolRefAttr":$symbol,
                          "mlir::FunctionType":$functionType);
    let cppNamespace = "::qcirc";
    let hasCustomAssemblyFormat = 1;
}

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_ATTRIBUTES_TD
