add_mlir_dialect_library(MLIRCCircTransforms
  CCircToQCircConversionPass.cpp

  DEPENDS
  MLIRCCircPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRCCirc
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

