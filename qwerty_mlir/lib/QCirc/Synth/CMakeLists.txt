add_mlir_dialect_library(MLIRQCircSynth
  SynthPerm.cpp
  SynthBennett.cpp

  LINK_LIBS PUBLIC
  MLIRQCirc
  MLIRCCirc
  MLIRArithDialect
)

