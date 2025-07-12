set(LLVM_TARGET_DEFINITIONS QwertyPasses.td)
mlir_tablegen(QwertyPasses.h.inc -gen-pass-decls -name Qwerty)
add_public_tablegen_target(MLIRQwertyPassIncGen)
add_dependencies(mlir-headers MLIRQwertyPassIncGen)
