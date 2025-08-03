add_mlir_dialect_library(MLIRQwertyTransforms
  SynthEmbedsPass.cpp
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
  MLIRQCirc
  MLIRCCirc
  MLIRPass
  MLIRTransforms
  # For qwerty/tweedledum.cpp
  qwutil
)

