get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRQwerty
  MLIRQwertyTransforms
  MLIRQwertyTestAnalysis
  MLIRQCirc
  MLIRQCircTransforms
)

add_llvm_executable(qwerty-opt
  qwerty-opt.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(qwerty-opt
  PRIVATE
  ${LIBS}
)

llvm_update_compile_flags(qwerty-opt)

install(TARGETS qwerty-opt)
