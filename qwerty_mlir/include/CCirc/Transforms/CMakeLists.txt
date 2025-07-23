set(LLVM_TARGET_DEFINITIONS CCircPasses.td)
mlir_tablegen(CCircPasses.h.inc -gen-pass-decls -name CCirc)
add_public_tablegen_target(MLIRCCircPassIncGen)
add_dependencies(mlir-headers MLIRCCircPassIncGen)
