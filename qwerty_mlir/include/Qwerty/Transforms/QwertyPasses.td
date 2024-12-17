//===-- QwertyPasses.td - Qwerty transformation definition file -*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_TRANSFORMS_QWERTY_PASSES_TD
#define DIALECT_INCLUDE_QWERTY_TRANSFORMS_QWERTY_PASSES_TD

include "mlir/Pass/PassBase.td"

def LiftLambdas : Pass<"lift-lambdas", "::mlir::ModuleOp"> {
  let summary = "Convert qwerty.lambda ops to qwerty.func ops";
  let constructor = "qwerty::createLiftLambdasPass()";
}

def OnlyPredOnes : Pass<"only-pred-ones"> {
  let summary = "Convert all func_pred ops to predicate only on 1s";
  let constructor = "qwerty::createOnlyPredOnesPass()";
}

def InlinePred : Pass<"inline-pred"> {
  let summary = "Inline pred ops";
  let constructor = "qwerty::createInlinePredPass()";
}

def QwertyToQCircConversion : Pass<"convert-qwerty-to-qcirc", "::mlir::ModuleOp"> {
  let summary = "Convert Qwerty dialect to QCirc dialect";
  let constructor = "qwerty::createQwertyToQCircConversionPass()";
  let dependentDialects = ["qcirc::QCircDialect", "mlir::scf::SCFDialect", "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}

#endif // DIALECT_INCLUDE_QWERTY_TRANSFORMS_QWERTY_PASSES_TD

