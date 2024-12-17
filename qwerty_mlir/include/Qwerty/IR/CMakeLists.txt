# Yanked from the arith dialect CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS QwertyAttributes.td)
mlir_tablegen(QwertyOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(QwertyOpsEnums.cpp.inc -gen-enum-defs)
# Currently these don't generate anything (apparently enum attrs end up in the
# enums header file), but generating them anyway to avoid grief later
mlir_tablegen(QwertyOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=qwerty)
mlir_tablegen(QwertyOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=qwerty)

# Yanked from mlir/IR/CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS QwertyInterfaces.td)
mlir_tablegen(QwertyOpsTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(QwertyOpsTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(QwertyOpsOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(QwertyOpsOpInterfaces.cpp.inc -gen-op-interface-defs)

add_mlir_dialect(QwertyOps qwerty)
