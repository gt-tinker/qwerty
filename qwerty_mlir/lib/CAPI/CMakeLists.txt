add_mlir_public_c_api_library(MLIRCAPIQCirc
  QCirc.cpp
  QCircPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRQCircPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQCirc
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPIQwerty
  Qwerty.cpp
  QwertyPasses.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  MLIRQwertyPassIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQwerty
  MLIRPass
)

add_mlir_public_c_api_library(MLIRCAPICCirc
  CCirc.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRCCirc
)

add_mlir_public_c_api_library(MLIRCAPIUtils
  Utils.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRCAPIExecutionEngine
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
)
