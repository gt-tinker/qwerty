set(LLVM_TARGET_DEFINITIONS QCircPasses.td)
mlir_tablegen(QCircPasses.h.inc -gen-pass-decls -name QCirc)
add_public_tablegen_target(MLIRQCircPassIncGen)
add_dependencies(mlir-headers MLIRQCircPassIncGen)
