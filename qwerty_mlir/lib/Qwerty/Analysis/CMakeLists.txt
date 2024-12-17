add_mlir_dialect_library(MLIRQwertyAnalysis
  DISABLE_INSTALL

  FuncSpecAnalysis.cpp
  QubitIndexAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRQwerty
)

