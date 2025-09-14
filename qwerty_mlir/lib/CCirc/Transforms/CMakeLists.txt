add_mlir_dialect_library(MLIRCCircTransforms
  CCircToXAGConversionPass.cpp

  DEPENDS
  MLIRCCircPassIncGen

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRCCirc
  MLIRPass
  MLIRTransforms
)

