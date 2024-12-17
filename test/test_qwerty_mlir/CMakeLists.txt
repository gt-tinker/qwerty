add_mlir_library(MLIRQwertyTestAnalysis
    DISABLE_INSTALL

    Qwerty/Analysis/TestFuncSpecAnalysisPass.cpp
    Qwerty/Analysis/TestQubitIndexAnalysisPass.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PUBLIC
    MLIRQwertyAnalysis
    MLIRQwerty
    MLIRPass
)
