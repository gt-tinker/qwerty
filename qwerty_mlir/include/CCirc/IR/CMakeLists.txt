# Yanked from the arith dialect CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS CCircAttributes.td)
# Currently these don't generate anything (apparently enum attrs end up in the
# enums header file), but generating them anyway to avoid grief later
mlir_tablegen(CCircOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=ccirc)
mlir_tablegen(CCircOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=ccirc)

add_mlir_dialect(CCircOps ccirc)
