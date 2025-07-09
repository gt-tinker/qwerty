get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRQCirc
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRToLLVMIRTranslationRegistration
  MLIRTargetLLVMIRExport
)

add_llvm_executable(qwerty-translate
  qwerty-translate.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(qwerty-translate
  PRIVATE
  ${LIBS}
)

llvm_update_compile_flags(qwerty-translate)

install(TARGETS qwerty-translate)
