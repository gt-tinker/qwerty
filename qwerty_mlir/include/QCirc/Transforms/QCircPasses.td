//===-- QCircPasses.td - QCirc transformation definition file -*- tablegen -*-===//

#ifndef DIALECT_INCLUDE_QCIRC_TRANSFORMS_QCIRC_PASSES_TD
#define DIALECT_INCLUDE_QCIRC_TRANSFORMS_QCIRC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DecomposeMultiControl : Pass<"decompose-multi-control"> {
  let summary = "Efficiently decompose multli-controlled gates";
  let constructor = "qcirc::createDecomposeMultiControlPass()";
}

def ReplaceAnnoyingGates : Pass<"replace-annoying-gates"> {
  let summary = "Replace Tweedledum IR gates with equivalents in QIR";
  let constructor = "qcirc::createReplaceAnnoyingGatesPass()";
}

def PeepholeOptimization : Pass<"peephole-optimization"> {
  let summary = "Replace specific gates to optimize the gates";
  let constructor = "qcirc::createPeepholeOptimizationPass()";
}

def InlineAdj : Pass<"inline-adj"> {
  let summary = "Replace every adj (adjoint) op with the adjoint of each of its containing gates in reverse order. This is mainly used for testing the AdjointBlock utility routines via qwerty-opt.";
  let constructor = "qcirc::createInlineAdjPass()";
}

def BaseProfileModulePrep : Pass<"base-profile-module-prep", "::mlir::ModuleOp"> {
  let summary = "Prepare module for lowering to QIR Base Profile";
  let constructor = "qcirc::createBaseProfileModulePrepPass()";
}

def BaseProfileFuncPrep : Pass<"base-profile-func-prep", "::mlir::func::FuncOp"> {
  let summary = "Prepare function for lowering to QIR Base Profile";
  let constructor = "qcirc::createBaseProfileFuncPrepPass()";
}

def QCircToQIRConversion : Pass<"convert-qcirc-to-qir", "::mlir::ModuleOp"> {
  let summary = "Lower QCirc dialect to QIR (LLVM IR)";
  let constructor = "qcirc::createQCircToQIRConversionPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // DIALECT_INCLUDE_QCIRC_TRANSFORMS_QCIRC_PASSES_TD

