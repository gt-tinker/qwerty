add_mlir_dialect_library(MLIRQCircUtils
  GenerateQasm.cpp
  AdjointBlock.cpp

  LINK_LIBS PUBLIC
  MLIRQCirc
  MLIRArithDialect
  MLIRFuncDialect
)

