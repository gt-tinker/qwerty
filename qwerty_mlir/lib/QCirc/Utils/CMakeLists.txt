add_mlir_dialect_library(MLIRQCircUtils
  GenerateQasm.cpp
  AdjointBlock.cpp
  SynthPerm.cpp
  SynthBennett.cpp

  LINK_LIBS PUBLIC
  MLIRQCirc
  MLIRCCirc
  MLIRArithDialect
  MLIRFuncDialect
)

