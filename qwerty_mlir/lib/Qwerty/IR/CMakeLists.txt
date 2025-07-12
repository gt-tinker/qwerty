add_mlir_dialect_library(MLIRQwerty
        DISABLE_INSTALL

        QwertyDialect.cpp
        QwertyOps.cpp
        QwertyInterfaces.cpp
        QwertyTypes.cpp
        QwertyAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRQwertyOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
    )
