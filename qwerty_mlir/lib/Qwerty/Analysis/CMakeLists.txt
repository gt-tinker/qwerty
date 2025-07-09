add_mlir_dialect_library(MLIRQwertyAnalysis
  FuncSpecAnalysis.cpp
  QubitIndexAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRQwerty
)

