add_mlir_dialect_library(MLIRCCircSynth
  SynthArith.cpp
  SynthRotate.cpp

  LINK_LIBS PUBLIC
  MLIRCCirc
)

