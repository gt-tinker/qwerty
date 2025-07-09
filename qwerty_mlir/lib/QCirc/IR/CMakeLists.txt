add_mlir_dialect_library(MLIRQCirc
        QCircDialect.cpp
        QCircOps.cpp
        QCircInterfaces.cpp
        QCircTypes.cpp
        QCircAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRQCircOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
    )
