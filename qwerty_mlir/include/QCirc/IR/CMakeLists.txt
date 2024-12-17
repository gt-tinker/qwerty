# Yanked from the arith dialect CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS QCircAttributes.td)
mlir_tablegen(QCircOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(QCircOpsEnums.cpp.inc -gen-enum-defs)
# Currently these don't generate anything (apparently enum attrs end up in the
# enums header file), but generating them anyway to avoid grief later
mlir_tablegen(QCircOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=qcirc)
mlir_tablegen(QCircOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=qcirc)

# Yanked from mlir/IR/CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS QCircInterfaces.td)
mlir_tablegen(QCircOpsTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(QCircOpsTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(QCircOpsOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(QCircOpsOpInterfaces.cpp.inc -gen-op-interface-defs)

add_mlir_dialect(QCircOps qcirc)
