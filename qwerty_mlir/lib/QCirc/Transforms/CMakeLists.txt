add_mlir_dialect_library(MLIRQCircTransforms
  DecomposeMultiControlPass.cpp
  ReplaceNonQIRGatesPass.cpp
  ReplaceNonQasmGatesPass.cpp
  InlineAdjPass.cpp
  PeepholeOptimizationPass.cpp
  BaseProfilePrepPasses.cpp
  QCircToQIRConversionPass.cpp

  DEPENDS
  MLIRQCircPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRQCirc
  MLIRPass
  MLIRTransforms
  MLIRNVVMDialect #we reference NVVM barrier ops
  MLIRAffineToStandard
  MLIRFuncInlinerExtension 
  MLIRSCFToControlFlow
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
)

