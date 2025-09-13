add_mlir_dialect_library(MLIRQCircUtils
  GenerateQasm.cpp
  AdjointBlock.cpp
  SynthPerm.cpp

  LINK_LIBS PUBLIC
  MLIRQCirc
  MLIRArithDialect
  MLIRFuncDialect
)

