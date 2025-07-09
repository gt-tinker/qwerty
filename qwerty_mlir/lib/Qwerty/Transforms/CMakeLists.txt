add_mlir_dialect_library(MLIRQwertyTransforms
  LiftLambdasPass.cpp
  OnlyPredOnesPass.cpp
  QwertyToQCircConversionPass.cpp
  InlinePredPass.cpp

  DEPENDS
  MLIRQwertyPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRFuncInlinerExtension 
  MLIRQwerty
  MLIRQwertyAnalysis
  MLIRQwertyUtils
  MLIRPass
  MLIRTransforms
  # For qwerty/tweedledum.cpp
  qwutil
)

