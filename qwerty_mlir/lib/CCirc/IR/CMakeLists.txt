add_mlir_dialect_library(MLIRCCirc

        CCircDialect.cpp
        CCircOps.cpp
        CCircTypes.cpp
        CCircAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRCCircOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
    )
