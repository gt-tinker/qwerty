add_mlir_public_c_api_library(MLIRCAPIQCirc
  QCirc.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQCirc
)

add_mlir_public_c_api_library(MLIRCAPIQwerty
  Qwerty.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQwerty
)
