add_mlir_dialect_library(MLIRQCircUtils
  DISABLE_INSTALL

  GenerateQasm.cpp
  AdjointBlock.cpp

  LINK_LIBS PUBLIC
  MLIRQCirc
  MLIRArithDialect
  MLIRFuncDialect
)

