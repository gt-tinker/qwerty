//===-- CCircPasses.td - CCirc transformation definition file -*- tablegen -*-===//

#ifndef DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_PASSES_TD
#define DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_PASSES_TD

include "mlir/Pass/PassBase.td"

def CCircToXAGConversion : Pass<"convert-ccirc-to-xag", "::ccirc::CircuitOp"> {
  let summary = "Convert the contents of every ccirc.circuit circuit op into a high-level XAG";
  let constructor = "ccirc::createCCircToXAGConversionPass()";
}

#endif // DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_PASSES_TD

