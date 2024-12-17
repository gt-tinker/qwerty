// This module is the harness connecting the Python frontend to the C++ guts of
// the compiler. Currently, it exposes a bare minimum of the C++ Qwerty AST
// data structure to Python in order for the Python frontend to walk the Python
// AST and generate a Qwerty AST.

// Per the documentation, Python.h needs to be the first #include
#define PY_SSIZE_T_CLEAN
// Defined by scikit-build-core based on the py-api in pyproject.toml
// ...but check, just in case
#ifndef Py_LIMITED_API
    #define Py_LIMITED_API 0x030A0000 /* >= 3.10 */
#endif
#include <Python.h>

#include "ast.hpp"
#include "mlir_handle.hpp"

// Needed because we define a variable named TupleType below
using QwertyTupleType = TupleType;

// This is not part of the stable ABI yet, so define it ourselves
static inline PyObject *PyObject_CallOneArg(PyObject *func, PyObject *arg) {
    PyObject *singleton_tuple = PyTuple_New(1);
    if (!singleton_tuple) {
        return NULL;
    }
    Py_INCREF(arg);
    if (PyTuple_SetItem(singleton_tuple, 0, arg) < 0) {
        Py_DECREF(arg);
        Py_DECREF(singleton_tuple);
        return NULL;
    }
    PyObject *ret = PyObject_Call(func, singleton_tuple, NULL);
    Py_DECREF(singleton_tuple);
    return ret;
}

///////////////////////// CONVERSIONS /////////////////////////

// Need to forward declare these since they use KernelType, which is not defined yet
static int convert_borrowed_hybrid_list(PyObject *hybrids_list, void *result_vector);
static int convert_owned_hybrid_list(PyObject *hybrids_list, void *result_vector);
// Uses ASTNodeType, which is not defined yet
static int convert_node_list(PyObject *stmts_list, void *result_vector);
// Uses DimVarExprType, which is not defined yet
static int convert_dimvarexpr_list(PyObject *dimvarexpr_list, void *result_vector);
// Uses TypeType, which is not defined yet
static int convert_type_list(PyObject *types_list, void *result_vector);

// The other conversion functions can be defined right now:

static int convert_dimvar_value_list(PyObject *dvv_list, void *result_vector) {
    std::vector<std::optional<DimVarValue>> *output_vector =
            (std::vector<std::optional<DimVarValue>> *)result_vector;
    PyObject *iter = NULL;
    PyObject *dvv = NULL;
    if (!(iter = PyObject_GetIter(dvv_list))) {
        goto failure;
    }
    while ((dvv = PyIter_Next(iter))) {
        if (dvv == Py_Ellipsis) {
            output_vector->push_back(std::nullopt);
        } else {
            if (!PyLong_Check(dvv)) {
                // TODO: make this more generic if we use this elsewhere
                PyErr_SetString(PyExc_ValueError, "Need to pass a list of ints to register_explicit_dimvars()");
                goto failure;
            }

            static_assert(std::is_same<ssize_t, DimVarValue>::value
                          && std::is_same<ssize_t, Py_ssize_t>::value,
                          "Conversion from Python int to DimVarValue needs to be updated");
            DimVarValue val = PyLong_AsSsize_t(dvv);
            if (val == -1 && PyErr_Occurred()) {
                goto failure;
            }
            output_vector->emplace_back(val);
        }

        Py_XDECREF(dvv);
        dvv = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(dvv);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

static int convert_size_t_list(PyObject *size_t_list, void *result_vector) {
    std::vector<size_t> *output_vector = (std::vector<size_t> *)result_vector;
    PyObject *iter = NULL;
    PyObject *size = NULL;
    if (!(iter = PyObject_GetIter(size_t_list))) {
        goto failure;
    }
    while ((size = PyIter_Next(iter))) {
        if (!PyLong_Check(size)) {
            PyErr_SetString(PyExc_ValueError, "Need to pass a list of unsigned ints");
            goto failure;
        }

        size_t val = PyLong_AsSize_t(size);
        if (val == (size_t)-1 && PyErr_Occurred()) {
            goto failure;
        }
        output_vector->push_back(val);

        Py_XDECREF(size);
        size = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(size);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

static int convert_string_list(PyObject *strs_list, void *result_vector) {
    std::vector<std::string> *output_vector = (std::vector<std::string> *)result_vector;
    PyObject *iter = NULL;
    PyObject *str = NULL;
    if (!(iter = PyObject_GetIter(strs_list))) {
        goto failure;
    }
    while ((str = PyIter_Next(iter))) {
        if (!PyUnicode_Check(str)) {
            // TODO: make this more generic if we use this elsewhere
            PyErr_SetString(PyExc_ValueError, "Need to pass strs as strs to Kernel constructor");
            goto failure;
        }

        const char *str_str = PyUnicode_AsUTF8AndSize(str, NULL);
        if (!str_str) {
            goto failure;
        }
        output_vector->emplace_back(str_str);

        Py_XDECREF(str);
        str = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(str);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}


// These macros are crazy, but I can think of only the following undesirable
// alternatives:
// 1. Hardcode everything, making this file extremely tedious
// 2. Use something like pybind11 (no external deps, please)
// 3. Use a tablegen-like tool similar to the one in CPython that generates the
//    extension code from some kind of specification (too complicated)
//
// These macros seem like the lesser evil to me, since the code using these
// macros is more meaningful and much more compact (>2.5x fewer lines). It's
// almost like a super-mini-DSL for Python types inside of C++. Oh, God...
#define AST_NODE_OBJECT_STRUCT(name, parent_name) \
    typedef struct name ## Object { \
        parent_name ## Object parent; \
    } name ## Object;

#define AST_NODE_ALLOC_FREE(name) \
    static PyObject *name ## _alloc(PyTypeObject *cls, Py_ssize_t nitems) { \
        assert(!nitems && #name " has no items"); \
        return PyObject_Init((PyObject *)new name ## Object(), cls); \
    } \
    static void name ## _free(name ## Object *self) { \
        delete self; \
    }

#define AST_NODE_INIT(name, parent_name) \
    static int name ## _subinit(name ## Object *self, PyObject *args); \
    static int name ## _init(name ## Object *self, PyObject *args, PyObject *kwds) { \
        PyObject *super_kwds = PyDict_New(); \
        if (!super_kwds) { \
            return -1; \
        } \
        PyObject *super_args = PyTuple_New(0); \
        if (!super_args) { \
            Py_DECREF(super_kwds); \
            return -1; \
        } \
        if (parent_name ## _init((parent_name ## Object *) self, super_args, super_kwds) < 0) { \
            Py_DECREF(super_kwds); \
            Py_DECREF(super_args); \
            return -1; \
        } \
        Py_DECREF(super_kwds); \
        Py_DECREF(super_args); \
        \
        (void)kwds; \
        return name ## _subinit(self, args); \
    }

// Need this because of the slots pfunc field not being a const pointer
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#define AST_NODE_TYPE_GLOBAL_VAR(name, more_slots) \
    static PyObject *name ## Type; \
    static PyObject *name ## _get_type(void) { \
        PyType_Slot spec_slots[] = { \
            {Py_tp_doc, (void *)PyDoc_STR(#name)}, \
            {Py_tp_init, (void *)name ## _init}, \
            {Py_tp_alloc, (void *)name ## _alloc}, \
            {Py_tp_new, (void *)PyType_GenericNew}, \
            {Py_tp_free, (void *)name ## _free}, \
            more_slots \
            {0, NULL} \
        }; \
        PyType_Spec spec = { \
            "_qwerty_harness." #name, /* name */ \
            sizeof (name ## Object), /* basicsize */ \
            0, /* itemsize */ \
            Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* flags */ \
            spec_slots /* slots */ \
        }; \
        PyObject *ret = PyType_FromSpec(&spec); \
        return ret; \
    }

#define SLOT_BASE(base) {Py_tp_base, (void *)base},
#define SLOT_NUMBER_METHODS(add, subtract, multiply) \
    {Py_nb_inplace_add, (void *)add}, \
    {Py_nb_inplace_subtract, (void *)subtract}, \
    {Py_nb_inplace_multiply, (void *)multiply},
#define SLOT_METHODS(methods) {Py_tp_methods, (void *)methods},

#define AST_NODE_BOILERPLATE(name, parent_name) \
    AST_NODE_OBJECT_STRUCT(name, parent_name) \
    AST_NODE_ALLOC_FREE(name) \
    AST_NODE_INIT(name, parent_name)

#define AST_NODE(name, parent_name) \
    AST_NODE_BOILERPLATE(name, parent_name) \
    AST_NODE_TYPE_GLOBAL_VAR(name, SLOT_BASE(parent_name ## Type))

#define AST_NODE_METHODS(name) \
    static PyMethodDef name ## _methods[] =

#define AST_NODE_TYPE_WITH_METHODS(name, parent_name) \
    AST_NODE_TYPE_GLOBAL_VAR(name, SLOT_BASE(parent_name ## Type) SLOT_METHODS(name ## _methods))

#define AST_NODE_TYPE_WITH_NUMBER_METHODS(name, parent_name, add, subtract, multiply) \
    AST_NODE_TYPE_GLOBAL_VAR(name, SLOT_BASE(parent_name ## Type) SLOT_NUMBER_METHODS(add, subtract, multiply))

#define AST_NODE_SUBINIT(name) \
    static int name ## _subinit(name ## Object *self, PyObject *args)

// Root versions
#define AST_ROOT_NODE_OBJECT_STRUCT(name) \
    typedef struct name ## Object { \
        PyObject_HEAD \
        std::unique_ptr<name> ptr; \
    } name ## Object;

#define AST_ROOT_NODE_INIT(name) \
    static int name ## _subinit(name ## Object *self, PyObject *args); \
    static int name ## _init(name ## Object *self, PyObject *args, PyObject *kwds) { \
        (void)kwds; \
        return name ## _subinit(self, args); \
    }

#define AST_ROOT_NODE_BOILERPLATE(name) \
    AST_ROOT_NODE_OBJECT_STRUCT(name) \
    AST_NODE_ALLOC_FREE(name) \
    AST_ROOT_NODE_INIT(name)

#define AST_ROOT_NODE(name) \
    AST_ROOT_NODE_BOILERPLATE(name) \
    AST_NODE_TYPE_GLOBAL_VAR(name, )

#define AST_ROOT_NODE_TYPE_WITH_METHODS(name) \
    AST_NODE_TYPE_GLOBAL_VAR(name, SLOT_METHODS(name ## _methods))

#define AST_ROOT_NODE_TYPE_WITH_NUMBER_METHODS(name, add, subtract, multiply) \
    AST_NODE_TYPE_GLOBAL_VAR(name, SLOT_NUMBER_METHODS(add, subtract, multiply))

#define AST_ROOT_NODE_TYPE_WITH_METHODS_AND_NUMBER_METHODS(name, add, subtract, multiply) \
    AST_NODE_TYPE_GLOBAL_VAR(name, SLOT_NUMBER_METHODS(add, subtract, multiply) SLOT_METHODS(name ## _methods))

#define AST_NODE_RUNTIME_SETUP(name) \
    if (!(name ## Type = name ## _get_type()) \
        || PyModule_AddObjectRef(m, #name, name ## Type) < 0) {  \
        return NULL; \
    }

///////////////////////// DEBUGINFO /////////////////////////

AST_ROOT_NODE_BOILERPLATE(DebugInfo)
AST_NODE_SUBINIT(DebugInfo) {
    if (!PyTuple_Size(args)) {
        // Used elsewhere in this file to create an empty DebugInfo that we
        // poke inside of to set the ptr
        return 0;
    // Copy constructor (kinda)
    } else if (PyTuple_Size(args) == 1) {
        DebugInfoObject *dbg;
        if (!PyArg_ParseTuple(args, "O!", Py_TYPE(self), &dbg)) {
            return -1;
        }
        self->ptr = std::move(dbg->ptr->copy());
        return 0;
    } else {
        const char *srcfile;
        unsigned int row;
        unsigned int col;
        PyObject *frame;
        if (!PyArg_ParseTuple(args, "sIIO", &srcfile, &row, &col, &frame)) {
            return -1;
        }
        if (frame == Py_None) {
            frame = NULL;
        }
        // Increments refcount of frame
        self->ptr = std::make_unique<DebugInfo>(srcfile, row, col, frame);
        return 0;
    }
}

static PyObject *DebugInfo_copy(DebugInfoObject *self, PyObject *args) {
    (void)args;
    DebugInfoObject *dbg = (DebugInfoObject *)PyObject_CallOneArg(
            (PyObject *)Py_TYPE(self), (PyObject *)self);
    if (!dbg) {
        return NULL;
    }
    return (PyObject *)dbg;
}

static PyObject *DebugInfo_get_frame(DebugInfoObject *self, PyObject *args) {
    (void)args;
    PyObject *frame = (PyObject *)self->ptr->python_frame;
    if (!frame) {
        frame = Py_None;
    }
    Py_INCREF(frame);
    return frame;
}

static PyObject *DebugInfo_get_row(DebugInfoObject *self, PyObject *args) {
    (void)args;
    return PyLong_FromUnsignedLong(self->ptr->row);
}

static PyObject *DebugInfo_get_col(DebugInfoObject *self, PyObject *args) {
    (void)args;
    return PyLong_FromUnsignedLong(self->ptr->col);
}

AST_NODE_METHODS(DebugInfo) {
    {"copy", (PyCFunction)DebugInfo_copy, METH_NOARGS, "Duplicate myself"},
    {"get_frame", (PyCFunction)DebugInfo_get_frame, METH_NOARGS, "Return either a Frame or None"},
    {"get_row", (PyCFunction)DebugInfo_get_row, METH_NOARGS, "Return line number"},
    {"get_col", (PyCFunction)DebugInfo_get_col, METH_NOARGS, "Return column number"},
    {NULL}
};

AST_ROOT_NODE_TYPE_WITH_METHODS(DebugInfo)

///////////////////////// EXCEPTION HANDLING /////////////////////////
// (Intentionally placed after DebugInfo so it can muck around with it)

static PyObject *QwertyProgrammerError;
static PyObject *QwertyTypeError;
static PyObject *QwertyCompileError;
// Not a QwertyProgrammerError because it doesn't have a frame
static PyObject *QwertyJITError;

// Implemented here to avoid calling the Python API directly inside the compiler
void DebugInfo::python_incref(void *p) { Py_XINCREF(p); }
void DebugInfo::python_decref(void *p) { Py_XDECREF(p); }

// Throw a subclass of QwertyProgrammerError and attach the supplied DebugInfo
// to it. (Hopefully) the Python code calling this will catch it and monkey
// with the stack trace to use the frame stored in the DebugInfo in the
// exception. (See err.py)
static void qwerty_programmer_error(PyObject *err_type, std::string &message, DebugInfo &dbg) {
    DebugInfoObject *dbg_obj = (DebugInfoObject *)PyObject_CallNoArgs(DebugInfoType);
    if (!dbg_obj) {
        // Leave the error indicator with this error message
        return;
    }
    dbg_obj->ptr = dbg.copy();

    // Augment message with line number since we do not currently have a way to
    // highlight particular portions of a line in our error messages (see err.py)
    std::string aug_message = message + " (at column " + std::to_string(dbg_obj->ptr->col) + ")";

    PyObject *msg_str, *exc;
    if (!(msg_str = PyUnicode_FromStringAndSize(aug_message.data(),
                                                aug_message.size()))
        || !(exc = PyObject_CallOneArg(err_type, msg_str))
        || PyObject_SetAttrString(exc, "dbg", (PyObject *)dbg_obj) < 0) {
        // Leave the error indicator as-is, with this error message
        return;
    }

    PyErr_SetObject(err_type, exc);
}

///////////////////////// MLIR HARNESS /////////////////////////

AST_ROOT_NODE_BOILERPLATE(MlirHandle)
AST_NODE_SUBINIT(MlirHandle) {
    const char *srcfile;
    if (!PyArg_ParseTuple(args, "s", &srcfile)) {
        return -1;
    }
    self->ptr = std::move(std::make_unique<MlirHandle>(srcfile));
    return 0;
}

static PyObject *MlirHandle_dump_module_ir(MlirHandleObject *self, PyObject *args) {
    (void)args;
    std::string module_ir_str = self->ptr->dump_module_ir();
    return PyUnicode_FromString(module_ir_str.c_str());
}

static PyObject *MlirHandle_dump_qir(MlirHandleObject *self, PyObject *args) {
    int to_base_profile;
    if (!PyArg_ParseTuple(args, "p", &to_base_profile)) {
        return NULL;
    }
    std::string llvm_ir_str = self->ptr->dump_qir(to_base_profile);
    return PyUnicode_FromString(llvm_ir_str.c_str());
}

static PyObject *MlirHandle_jit(MlirHandleObject *self, PyObject *args) {
    (void)args;
    try {
        self->ptr->jit();
    } catch (JITException &err) {
        PyErr_SetString(QwertyJITError, err.message.c_str());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *MlirHandle_set_func_opt(MlirHandleObject *self, PyObject *args) {
    int do_func_opt;
    if (!PyArg_ParseTuple(args, "p", &do_func_opt)) {
        return NULL;
    }
    self->ptr->set_func_opt(do_func_opt);
    Py_RETURN_NONE;
}

AST_NODE_METHODS(MlirHandle) {
    {"dump_qir", (PyCFunction)MlirHandle_dump_qir, METH_VARARGS, "Return QIR as str"},
    {"dump_module_ir", (PyCFunction)MlirHandle_dump_module_ir, METH_NOARGS, "Return MLIR Module IR as a str"},
    {"jit", (PyCFunction)MlirHandle_jit, METH_NOARGS, "Perform optimizations and JIT for local simulation"},
    {"set_func_opt", (PyCFunction)MlirHandle_set_func_opt, METH_VARARGS, "For experimentation: turn function-related optimizations on or off"},
    {NULL}
};
AST_ROOT_NODE_TYPE_WITH_METHODS(MlirHandle)

///////////////////////// DIMVAREXPR /////////////////////////

AST_ROOT_NODE_BOILERPLATE(DimVarExpr)
AST_NODE_SUBINIT(DimVarExpr) {
    if (PyTuple_Size(args) == 1) {
        DimVarExprObject *dimvar_expr;
        if (!PyArg_ParseTuple(args, "O!", Py_TYPE(self), &dimvar_expr)) {
            return -1;
        }
        self->ptr = std::move(dimvar_expr->ptr->copy());
        return 0;
    } else {
        const char *dimvar;
        unsigned int offset;
        if (!PyArg_ParseTuple(args, "sI", &dimvar, &offset)) {
            return -1;
        }
        self->ptr = std::make_unique<DimVarExpr>(dimvar, offset);
        return 0;
    }
}

static PyObject *DimVarExpr_iadd(DimVarExprObject *self, DimVarExprObject *other) {
    if (Py_TYPE(self) != Py_TYPE(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    *self->ptr += *other->ptr;
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *DimVarExpr_isubtract(DimVarExprObject *self, DimVarExprObject *other) {
    if (Py_TYPE(self) != Py_TYPE(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    *self->ptr -= *other->ptr;
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *DimVarExpr_imultiply(DimVarExprObject *self, DimVarExprObject *other) {
    if (Py_TYPE(self) != Py_TYPE(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    *self->ptr *= *other->ptr;
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *DimVarExpr_is_constant(DimVarExprObject *self, PyObject *args) {
    (void)args;
    return PyBool_FromLong(self->ptr->isConstant());
}

static PyObject *DimVarExpr_get_value(DimVarExprObject *self, PyObject *args) {
    (void)args;

    if (!self->ptr->isConstant()) {
        PyErr_SetString(PyExc_ValueError, "DimVarExpr contains dimvars. "
                                          "Cannot return numeric value.");
        return NULL;
    }

    static_assert(std::is_same<Py_ssize_t, DimVarValue>::value,
                  "Conversion from DimVarValue to Python int needs to be "
                  "updated");
    return PyLong_FromSsize_t(self->ptr->offset);
}

static PyObject *DimVarExpr_copy(DimVarExprObject *self, PyObject *args) {
    DimVarExprObject *dimvar_expr = (DimVarExprObject *)PyObject_CallOneArg(
            (PyObject *)Py_TYPE(self), (PyObject *)self);
    if (!dimvar_expr) {
        return NULL;
    }
    return (PyObject *)dimvar_expr;
}

AST_NODE_METHODS(DimVarExpr) {
    {"is_constant", (PyCFunction)DimVarExpr_is_constant, METH_NOARGS, "Return whether this DimVarExpr is free of dimvars"},
    {"get_value", (PyCFunction)DimVarExpr_get_value, METH_NOARGS, "If this DimVarExpr contains no dimvars, return its integer value"},
    {"copy", (PyCFunction)DimVarExpr_copy, METH_NOARGS, "Return a copy of this DimVarExpr"},
    {NULL}
};

AST_ROOT_NODE_TYPE_WITH_METHODS_AND_NUMBER_METHODS(DimVarExpr, DimVarExpr_iadd, DimVarExpr_isubtract, DimVarExpr_imultiply)

static int convert_dimvarexpr_list(PyObject *dimvarexpr_list, void *result_vector) {
    std::vector<std::unique_ptr<DimVarExpr>> *output_vector =
            (std::vector<std::unique_ptr<DimVarExpr>> *)result_vector;
    PyObject *iter = NULL;
    PyObject *dimvarexpr_obj = NULL;
    if (!(iter = PyObject_GetIter(dimvarexpr_list))) {
        goto failure;
    }
    while ((dimvarexpr_obj = PyIter_Next(iter))) {
        if (!Py_IS_TYPE(dimvarexpr_obj, (PyTypeObject *)DimVarExprType)) {
            PyErr_SetString(PyExc_ValueError, "Expected list of DimVarExprs. What is this?!");
            goto failure;
        }
        DimVarExprObject *dimvar_expr = (DimVarExprObject *)dimvarexpr_obj;
        output_vector->push_back(std::move(dimvar_expr->ptr));

        Py_XDECREF(dimvarexpr_obj);
        dimvarexpr_obj = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(dimvarexpr_obj);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

///////////////////////// TYPE /////////////////////////

AST_ROOT_NODE_BOILERPLATE(Type)
AST_NODE_SUBINIT(Type) { return 0; }

static PyObject *Type_copy(TypeObject *self, PyObject *args) {
    (void)args;
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs((PyObject *)Py_TYPE(self));
    if (!type) {
        return NULL;
    }
    type->ptr = std::move(self->ptr->copy());
    return (PyObject *)type;
}

static PyObject *Type_as_embedded(TypeObject *self, PyObject *args) {
    DebugInfoObject *dbg;
    EmbeddingKind kind;
    if (!PyArg_ParseTuple(args, "O!i",
                          DebugInfoType, &dbg,
                          &kind)) {
        return NULL;
    }

    TypeObject *type = (TypeObject *)PyObject_CallNoArgs((PyObject *)Py_TYPE(self));
    if (!type) {
        return NULL;
    }

    try {
        type->ptr = std::move(self->ptr->asEmbedded(*dbg->ptr, kind));
    } catch (TypeException &err) {
        qwerty_programmer_error(QwertyTypeError, err.message, *err.dbg);
        return NULL;
    }

    return (PyObject *)type;
}

static PyObject *Type_new_qubit(PyObject *cls, PyObject *args) {
    DimVarExprObject *dimvar_expr;
    if (!PyArg_ParseTuple(args, "O!", DimVarExprType, (PyObject *)&dimvar_expr)) {
        return NULL;
    }

    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::make_unique<QubitType>(std::move(dimvar_expr->ptr));
    return (PyObject *)type;
}

static PyObject *Type_new_bit(PyObject *cls, PyObject *args) {
    DimVarExprObject *dimvar_expr;
    if (!PyArg_ParseTuple(args, "O!", DimVarExprType, (PyObject *)&dimvar_expr)) {
        return NULL;
    }

    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::make_unique<BitType>(std::move(dimvar_expr->ptr));
    return (PyObject *)type;
}

static PyObject *Type_new_int(PyObject *cls, PyObject *args) {
    (void)args;
    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::make_unique<IntType>();
    return (PyObject *)type;
}

static PyObject *Type_new_float(PyObject *cls, PyObject *args) {
    (void)args;
    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::make_unique<FloatType>();
    return (PyObject *)type;
}

static PyObject *Type_new_tuple(PyObject *cls, PyObject *args) {
    std::unique_ptr<QwertyTupleType> tuple;
    if (!PyTuple_Size(args)) { // Unit
        tuple = std::make_unique<QwertyTupleType>();
    } else {
        std::vector<std::unique_ptr<Type>> types;
        if (!PyArg_ParseTuple(args, "O&", convert_type_list, (void *)&types)) {
            return NULL;
        }
        tuple = std::make_unique<QwertyTupleType>(std::move(types));
    }

    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::move(tuple);
    return (PyObject *)type;
}

static PyObject *Type_new_func_common(PyObject *cls, PyObject *args, bool is_rev) {
    TypeObject *lhs;
    TypeObject *rhs;
    // This is a @classmethod, so cls is the Type type object
    if (!PyArg_ParseTuple(args, "O!O!", cls, &lhs, cls, &rhs)) {
        return NULL;
    }

    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::make_unique<FuncType>(std::move(lhs->ptr),
                                           std::move(rhs->ptr),
                                           is_rev);
    return (PyObject *)type;
}

static PyObject *Type_new_func(PyObject *cls, PyObject *args) {
    return Type_new_func_common(cls, args, false);
}

static PyObject *Type_new_rev_func(PyObject *cls, PyObject *args) {
    return Type_new_func_common(cls, args, true);
}

static PyObject *Type_new_broadcast(PyObject *cls, PyObject *args) {
    TypeObject *elem_type;
    DimVarExprObject *factor;
    if (!PyArg_ParseTuple(args, "O!O!",
                          cls, &elem_type,
                          DimVarExprType, &factor)) {
        return NULL;
    }

    // This is a @classmethod, so cls is the Type type object
    TypeObject *type = (TypeObject *)PyObject_CallNoArgs(cls);
    if (!type) {
        return NULL;
    }
    type->ptr = std::make_unique<BroadcastType>(std::move(elem_type->ptr),
                                                std::move(factor->ptr));
    return (PyObject *)type;
}

static PyObject *Type_bit_partitions(TypeObject *self, PyObject *args) {
    (void)args;

    Type *type = self->ptr.get();
    FuncType *func_type;

    if (!(func_type = dynamic_cast<FuncType *>(type))) {
        // Throw an error for this case instead of returning None because this
        // is almost certainly a bug
        PyErr_SetString(PyExc_ValueError,
                        "bit_partitions() must be called on a FuncType");
        return NULL;
    }

    type = func_type->lhs.get();

    if (BitType *bit_type = dynamic_cast<BitType *>(type)) {
        if (!bit_type->isConstant()) {
            Py_RETURN_NONE;
        }

        PyObject *list = PyList_New(1);
        if (!list) {
            return NULL;
        }
        if (PyList_SetItem(list, 0, PyLong_FromLong(bit_type->dim->offset)) < 0) {
            return NULL;
        }
        return list;
    } else if (TupleType *tuple_type = dynamic_cast<TupleType *>(type)) {
        if (!tuple_type->isConstant()) {
            Py_RETURN_NONE;
        }

        PyObject *list = PyList_New(tuple_type->types.size());
        if (!list) {
            return NULL;
        }
        for (size_t i = 0; i < tuple_type->types.size(); i++) {
            Type *elem_type = tuple_type->types[i].get();
            if (BitType *bit_type = dynamic_cast<BitType *>(elem_type)) {
                if (PyList_SetItem(list, i, PyLong_FromLong(bit_type->dim->offset)) < 0) {
                    Py_DECREF(list);
                    return NULL;
                }
            } else {
                Py_DECREF(list);
                Py_RETURN_NONE;
            }
        }
        return list;
    } else {
        Py_RETURN_NONE;
    }
}

AST_NODE_METHODS(Type) {
    {"copy", (PyCFunction)Type_copy, METH_NOARGS, "Duplicate myself"},
    {"as_embedded", (PyCFunction)Type_as_embedded, METH_VARARGS, "Return embedded version of this type"},
    {"bit_partitions", (PyCFunction)Type_bit_partitions, METH_NOARGS, "Calculate the dimension of each bit argument"},
    {"new_qubit", (PyCFunction)Type_new_qubit, METH_VARARGS | METH_CLASS, "Create a new qubit type"},
    {"new_bit", (PyCFunction)Type_new_bit, METH_VARARGS | METH_CLASS, "Create a new bit type"},
    {"new_int", (PyCFunction)Type_new_int, METH_NOARGS | METH_CLASS, "Create a new int type"},
    {"new_float", (PyCFunction)Type_new_float, METH_NOARGS | METH_CLASS, "Create a new float type"},
    {"new_tuple", (PyCFunction)Type_new_tuple, METH_VARARGS | METH_CLASS, "Create a new tuple type"},
    {"new_func", (PyCFunction)Type_new_func, METH_VARARGS | METH_CLASS, "Create a new func type"},
    {"new_rev_func", (PyCFunction)Type_new_rev_func, METH_VARARGS | METH_CLASS, "Create a new reversible func type"},
    {"new_broadcast", (PyCFunction)Type_new_broadcast, METH_VARARGS | METH_CLASS, "Create a new BroadcastType"},
    {NULL}
};

AST_ROOT_NODE_TYPE_WITH_METHODS(Type)

static int convert_type_list(PyObject *types_list, void *result_vector) {
    std::vector<std::unique_ptr<Type>> *output_vector =
            (std::vector<std::unique_ptr<Type>> *)result_vector;
    PyObject *iter = NULL;
    PyObject *type_obj = NULL;
    if (!(iter = PyObject_GetIter(types_list))) {
        goto failure;
    }
    while ((type_obj = PyIter_Next(iter))) {
        if (!Py_IS_TYPE(type_obj, (PyTypeObject *)TypeType)) {
            PyErr_SetString(PyExc_ValueError, "Expected list of Types. What is this?!");
            goto failure;
        }
        TypeObject *type = (TypeObject *)type_obj;
        output_vector->push_back(std::move(type->ptr));

        Py_XDECREF(type_obj);
        type_obj = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(type_obj);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

///////////////////////// HYBRID PYTHON-QWERTY OBJECTS /////////////////////////

AST_ROOT_NODE(HybridObj)
AST_NODE_SUBINIT(HybridObj) { return 0; }

///////////////////////// HYBRID BIT OBJECT /////////////////////////

AST_NODE_BOILERPLATE(Bits, HybridObj)
AST_NODE_SUBINIT(Bits) {
    // Special case: building this from a Bits pointer. We'll come back and
    // set the bits pointer later
    if (!PyTuple_Size(args)) {
        return 0;
    }

    PyObject *bytes;
    Py_ssize_t n_bits;
    if (!PyArg_ParseTuple(args, "O!n",
                          &PyBytes_Type, &bytes,
                          &n_bits)) {
        return -1;
    }

    Py_ssize_t full_bytes = PyBytes_Size(bytes);
    Py_ssize_t full_bits = full_bytes << 3;
    if (full_bits < n_bits) {
        PyErr_SetString(PyExc_ValueError, "bytes are too small for n_bits provided");
        return -1;
    }
    Py_ssize_t skip_bits = full_bits - n_bits;
    if (skip_bits >= 8) {
        PyErr_SetString(PyExc_ValueError, "bytes contains extra zero bytes (probably)");
        return -1;
    }
    char *str = PyBytes_AsString(bytes);
    if (!str) {
        return -1;
    }
    std::vector<bool> our_bits;
    for (Py_ssize_t i = 0; i < full_bytes; i++) {
        char c = str[i];
        Py_ssize_t j = (!i)? skip_bits : 0;
        for (; j < 8; j++) {
            our_bits.push_back((c >> (7-j)) & 0x1);
        }
    }

    self->parent.ptr = std::move(std::make_unique<Bits>(our_bits));
    return 0;
}

static PyObject *Bits_as_bytes(BitsObject *self, PyObject *args) {
    (void)args;
    Bits &bits = dynamic_cast<Bits &>(*self->parent.ptr);
    size_t n_bits = bits.getNumBits();
    // Round to next multiple of 8
    // (https://stackoverflow.com/a/1766566/321301), then divide by 8
    size_t n_bytes = ((n_bits + 0b111ULL) & ~0b111ULL) >> 3;
    size_t skip_bits = (n_bytes << 3) - n_bits;
    char *buf = new char[n_bytes+1]();

    for (size_t i = 0; i < n_bits; i++) {
        size_t i_off = i + skip_bits;
        size_t buf_idx = i_off >> 3;
        size_t byte_idx = 7 - (i_off & 0b111ULL);
        buf[buf_idx] |= ((char)bits.getBit(i) << byte_idx);
    }

    PyObject *ret = PyBytes_FromStringAndSize(buf, (Py_ssize_t)n_bytes);
    delete [] buf;
    return ret;
}

static PyObject *Bits_get_n_bits(BitsObject *self, PyObject *args) {
    (void)args;
    Bits &bits = dynamic_cast<Bits &>(*self->parent.ptr);
    return PyLong_FromSize_t(bits.getNumBits());
}

AST_NODE_METHODS(Bits) {
    {"as_bytes", (PyCFunction)Bits_as_bytes, METH_NOARGS, "Return bytes Python object ready to pass to int.from_bytes()"},
    {"get_n_bits", (PyCFunction)Bits_get_n_bits, METH_NOARGS, "Get number of bits"},
    {NULL}
};

AST_NODE_TYPE_WITH_METHODS(Bits, HybridObj)

///////////////////////// HYBRID INT OBJECT /////////////////////////

AST_NODE_BOILERPLATE(Integer, HybridObj)

AST_NODE_SUBINIT(Integer) {
    static_assert(std::is_same<Py_ssize_t, DimVarValue>::value,
                  "Conversion from Python int to DimVarValue needs to be updated");

    Py_ssize_t intval;
    if (!PyArg_ParseTuple(args, "n",
                          &intval)) {
        return -1;
    }

    self->parent.ptr = std::make_unique<Integer>(intval);
    return 0;
}

static PyObject *Integer_as_pyint(BitsObject *self, PyObject *args) {
    (void)args;
    Integer &int_ = dynamic_cast<Integer &>(*self->parent.ptr);

    static_assert(std::is_same<Py_ssize_t, DimVarValue>::value,
                  "Conversion from Python int to DimVarValue needs to be updated");
    return PyLong_FromSsize_t(int_.val);
}

AST_NODE_METHODS(Integer) {
    {"as_pyint", (PyCFunction)Integer_as_pyint, METH_NOARGS, "Return Python int"},
    {NULL}
};

AST_NODE_TYPE_WITH_METHODS(Integer, HybridObj)

///////////////////////// HYBRID ANGLE OBJECT /////////////////////////

AST_NODE_BOILERPLATE(Angle, HybridObj)

AST_NODE_SUBINIT(Angle) {
    double angle;
    if (!PyArg_ParseTuple(args, "d",
                          &angle)) {
        return -1;
    }

    self->parent.ptr = std::make_unique<Angle>(angle);
    return 0;
}

static PyObject *Angle_as_pyfloat(BitsObject *self, PyObject *args) {
    (void)args;
    Angle &angle = dynamic_cast<Angle &>(*self->parent.ptr);
    return PyFloat_FromDouble(angle.val);
}

AST_NODE_METHODS(Angle) {
    {"as_pyfloat", (PyCFunction)Angle_as_pyfloat, METH_NOARGS, "Return Python float"},
    {NULL}
};

AST_NODE_TYPE_WITH_METHODS(Angle, HybridObj)

///////////////////////// HYBRID TUPLE OBJECT /////////////////////////

AST_NODE(Tuple, HybridObj)
AST_NODE_SUBINIT(Tuple) {
    std::vector<std::unique_ptr<HybridObj>> children;
    if (!PyArg_ParseTuple(args, "O&",
                          convert_owned_hybrid_list, (void *)&children)) {
        return -1;
    }
    self->parent.ptr = std::make_unique<Tuple>(std::move(children));
    return 0;
}

///////////////////////// AST NODE /////////////////////////

AST_ROOT_NODE_BOILERPLATE(ASTNode)
AST_NODE_SUBINIT(ASTNode) { return 0; }

static PyObject *ASTNode_get_type(ASTNodeObject *self, PyObject *args) {
    (void)args;
    std::unique_ptr<Type> type = std::move(self->ptr->getType().copy());
    TypeObject *typeObj = (TypeObject *)PyObject_CallNoArgs(TypeType);
    if (!typeObj) {
        return NULL;
    }
    typeObj->ptr = std::move(type);
    return (PyObject *)typeObj;
}

static PyObject *ASTNode_get_dbg(ASTNodeObject *self, PyObject *args) {
    (void)args;
    std::unique_ptr<DebugInfo> dbg = std::move(self->ptr->dbg->copy());
    DebugInfoObject *dbgObj = (DebugInfoObject *)PyObject_CallNoArgs(DebugInfoType);
    if (!dbgObj) {
        return NULL;
    }
    dbgObj->ptr = std::move(dbg);
    return (PyObject *)dbgObj;
}

AST_NODE_METHODS(ASTNode) {
    {"get_type", (PyCFunction)ASTNode_get_type, METH_NOARGS, "Get type of this node"},
    {"get_dbg", (PyCFunction)ASTNode_get_dbg, METH_NOARGS, "Get DebugInfo of this node"},
    {NULL}
};

AST_ROOT_NODE_TYPE_WITH_METHODS(ASTNode)

static int convert_node_list(PyObject *stmts_list, void *result_vector) {
    std::vector<std::unique_ptr<ASTNode>> *output_vector =
            (std::vector<std::unique_ptr<ASTNode>> *)result_vector;
    PyObject *iter = NULL;
    PyObject *node_obj = NULL;
    if (!(iter = PyObject_GetIter(stmts_list))) {
        goto failure;
    }
    while ((node_obj = PyIter_Next(iter))) {
        int ret;
        if ((ret = PyObject_IsInstance(node_obj, ASTNodeType)) == -1) {
            goto failure;
        } else if (!ret) {
            PyErr_SetString(PyExc_ValueError, "Need to pass list of ASTNodes");
            goto failure;
        }
        ASTNodeObject *node = (ASTNodeObject *)node_obj;
        output_vector->push_back(std::move(node->ptr));

        Py_XDECREF(node_obj);
        node_obj = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(node_obj);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

///////////////////////// EXPRESSION /////////////////////////

AST_NODE(Expr, ASTNode)
AST_NODE_SUBINIT(Expr) { return 0; }

///////////////////////// VARIABLE /////////////////////////

AST_NODE(Variable, Expr)
AST_NODE_SUBINIT(Variable) {
    DebugInfoObject *dbg;
    const char *name;
    if (!PyArg_ParseTuple(args, "O!s",
                          DebugInfoType, &dbg,
                          &name)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Variable>(
            std::move(dbg->ptr), name);
    return 0;
}

///////////////////////// ADJOINT /////////////////////////

AST_NODE(Adjoint, Expr)
AST_NODE_SUBINIT(Adjoint) {
    DebugInfoObject *dbg;
    ExprObject *operand;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &operand)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Adjoint>(
            std::move(dbg->ptr), std::move(operand->parent.ptr));
    return 0;
}

///////////////////////// PREPARE QUBIT LITERAL/BITS /////////////////////////

AST_NODE(Prepare, Expr)
AST_NODE_SUBINIT(Prepare) {
    DebugInfoObject *dbg;
    ExprObject *operand;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &operand)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Prepare>(
            std::move(dbg->ptr), std::move(operand->parent.ptr));
    return 0;
}

///////////////////////// LIFT BITS TO QUBITS /////////////////////////

AST_NODE(LiftBits, Expr)
AST_NODE_SUBINIT(LiftBits) {
    DebugInfoObject *dbg;
    ExprObject *bits;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &bits)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<LiftBits>(
            std::move(dbg->ptr), std::move(bits->parent.ptr));
    return 0;
}

///////////////////////// EMBED A CLASSICAL FUNCTION /////////////////////////

AST_NODE(EmbedClassical, Expr)
AST_NODE_SUBINIT(EmbedClassical) {
    DebugInfoObject *dbg;
    const char *name, *operand_name;
    EmbeddingKind kind;
    if (!PyArg_ParseTuple(args, "O!ssi",
                          DebugInfoType, &dbg,
                          &name, &operand_name, &kind)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<EmbedClassical>(
            std::move(dbg->ptr), name, operand_name, kind);
    return 0;
}

///////////////////////// TENSOR PRODUCT WITH TWO OPERANDS /////////////////////////

AST_NODE(BiTensor, Expr)
AST_NODE_SUBINIT(BiTensor) {
    DebugInfoObject *dbg;
    ExprObject *left, *right;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &left,
                          ExprType, &right)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BiTensor>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(left->parent.ptr),
            unique_downcast<Expr, ASTNode>(right->parent.ptr));
    return 0;
}

///////////////////////// TENSOR PRODUCT WITH ONE OPERAND /////////////////////////

AST_NODE(BroadcastTensor, Expr)
AST_NODE_SUBINIT(BroadcastTensor) {
    DebugInfoObject *dbg;
    ExprObject *value;
    DimVarExprObject *factor;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &value,
                          DimVarExprType, &factor)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BroadcastTensor>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(value->parent.ptr),
            std::move(factor->ptr));
    return 0;
}

///////////////////////// VECTOR /////////////////////////

AST_NODE(QubitLiteral, Expr)
AST_NODE_SUBINIT(QubitLiteral) {
    DebugInfoObject *dbg;
    Eigenstate eigenstate;
    PrimitiveBasis prim_basis;
    DimVarExprObject *dim;
    if (!PyArg_ParseTuple(args, "O!iiO!",
                          DebugInfoType, &dbg,
                          &eigenstate, &prim_basis,
                          DimVarExprType, &dim)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<QubitLiteral>(
            std::move(dbg->ptr),
            eigenstate,
            prim_basis,
            std::move(dim->ptr));
    return 0;
}

///////////////////////// PHASE /////////////////////////

AST_NODE(Phase, Expr)
AST_NODE_SUBINIT(Phase) {
    DebugInfoObject *dbg;
    ExprObject *phase;
    ExprObject *value;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &phase,
                          ExprType, &value)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Phase>(
            std::move(dbg->ptr),
            std::move(phase->parent.ptr),
            std::move(value->parent.ptr));
    return 0;
}

///////////////////////// FLOAT LITERAL /////////////////////////

AST_NODE(FloatLiteral, Expr)
AST_NODE_SUBINIT(FloatLiteral) {
    DebugInfoObject *dbg;
    double value;
    if (!PyArg_ParseTuple(args, "O!d",
                          DebugInfoType, &dbg,
                          &value)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<FloatLiteral>(
            std::move(dbg->ptr),
            value);
    return 0;
}

///////////////////////// FLOAT NEGATE /////////////////////////

AST_NODE(FloatNeg, Expr)
AST_NODE_SUBINIT(FloatNeg) {
    DebugInfoObject *dbg;
    ExprObject *operand;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &operand)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<FloatNeg>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(operand->parent.ptr));
    return 0;
}

///////////////////////// FLOAT DIVIDE /////////////////////////

AST_NODE(FloatBinaryOp, Expr)
AST_NODE_SUBINIT(FloatBinaryOp) {
    DebugInfoObject *dbg;
    FloatOp op;
    ExprObject *left, *right;
    if (!PyArg_ParseTuple(args, "O!iO!O!",
                          DebugInfoType, &dbg,
                          &op,
                          ExprType, &left,
                          ExprType, &right)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<FloatBinaryOp>(
            std::move(dbg->ptr),
            op,
            unique_downcast<Expr, ASTNode>(left->parent.ptr),
            unique_downcast<Expr, ASTNode>(right->parent.ptr));
    return 0;
}

///////////////////////// FLOAT CONSTANT EXPRESSION /////////////////////////

AST_NODE(FloatDimVarExpr, Expr)
AST_NODE_SUBINIT(FloatDimVarExpr) {
    DebugInfoObject *dbg;
    DimVarExprObject *dim;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          DimVarExprType, &dim)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<FloatDimVarExpr>(
            std::move(dbg->ptr),
            std::move(dim->ptr));
    return 0;
}

///////////////////////// TUPLE LITERAL /////////////////////////

AST_NODE(TupleLiteral, Expr)
AST_NODE_SUBINIT(TupleLiteral) {
    DebugInfoObject *dbg;
    std::vector<std::unique_ptr<ASTNode>> elts;
    if (!PyArg_ParseTuple(args, "O!O&",
                          DebugInfoType, &dbg,
                          convert_node_list, (void *)&elts)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<TupleLiteral>(
            std::move(dbg->ptr),
            std::move(elts));
    return 0;
}

///////////////////////// STANDARD BASIS /////////////////////////

AST_NODE(BuiltinBasis, Expr)
AST_NODE_SUBINIT(BuiltinBasis) {
    DebugInfoObject *dbg;
    PrimitiveBasis prim_basis;
    DimVarExprObject *dim;
    if (!PyArg_ParseTuple(args, "O!iO!",
                          DebugInfoType, &dbg,
                          &prim_basis,
                          DimVarExprType, &dim)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BuiltinBasis>(
            std::move(dbg->ptr),
            prim_basis,
            std::move(dim->ptr));
    return 0;
}

///////////////////////// IDENTITY MAPPER /////////////////////////

AST_NODE(Identity, Expr)
AST_NODE_SUBINIT(Identity) {
    DebugInfoObject *dbg;
    DimVarExprObject *dim;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          DimVarExprType, &dim)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Identity>(
            std::move(dbg->ptr),
            std::move(dim->ptr));
    return 0;
}

///////////////////////// SUBSPACE TRANSFORM MAPPER /////////////////////////

AST_NODE(BasisTranslation, Expr)
AST_NODE_SUBINIT(BasisTranslation) {
    DebugInfoObject *dbg;
    ExprObject *basis_in;
    ExprObject *basis_out;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &basis_in,
                          ExprType, &basis_out)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BasisTranslation>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(basis_in->parent.ptr),
            unique_downcast<Expr, ASTNode>(basis_out->parent.ptr));
    return 0;
}

///////////////////////// DISCARDING /////////////////////////

AST_NODE(Discard, Expr)
AST_NODE_SUBINIT(Discard) {
    DebugInfoObject *dbg;
    DimVarExprObject *dim;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          DimVarExprType, &dim)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Discard>(std::move(dbg->ptr),
                                                        std::move(dim->ptr));
    return 0;
}

///////////////////////// MEASUREMENT /////////////////////////

AST_NODE(Measure, Expr)
AST_NODE_SUBINIT(Measure) {
    DebugInfoObject *dbg;
    ExprObject *basis;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &basis)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Measure>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(basis->parent.ptr));
    return 0;
}

///////////////////////// PROJECTION /////////////////////////

AST_NODE(Project, Expr)
AST_NODE_SUBINIT(Project) {
    DebugInfoObject *dbg;
    ExprObject *basis;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &basis)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Project>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(basis->parent.ptr));
    return 0;
}

///////////////////////// FLIP A BASIS /////////////////////////

AST_NODE(Flip, Expr)
AST_NODE_SUBINIT(Flip) {
    DebugInfoObject *dbg;
    ExprObject *basis;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &basis)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Flip>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(basis->parent.ptr));
    return 0;
}

///////////////////////// ROTATE AROUND A BASIS /////////////////////////

AST_NODE(Rotate, Expr)
AST_NODE_SUBINIT(Rotate) {
    DebugInfoObject *dbg;
    ExprObject *basis, *theta;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &basis,
                          ExprType, &theta)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Rotate>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(basis->parent.ptr),
            unique_downcast<Expr, ASTNode>(theta->parent.ptr));
    return 0;
}

///////////////////////// BASIS LIST /////////////////////////

AST_NODE(BasisLiteral, Expr)
AST_NODE_SUBINIT(BasisLiteral) {
    DebugInfoObject *dbg;
    std::vector<std::unique_ptr<ASTNode>> basis_elts;
    if (!PyArg_ParseTuple(args, "O!O&",
                          DebugInfoType, &dbg,
                          convert_node_list, (void *)&basis_elts)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BasisLiteral>(
            std::move(dbg->ptr),
            std::move(basis_elts));
    return 0;
}

///////////////////////// CONDITIONAL EXPRESSION /////////////////////////

AST_NODE(Conditional, Expr)
AST_NODE_SUBINIT(Conditional) {
    DebugInfoObject *dbg;
    ExprObject *if_expr;
    ExprObject *then_expr;
    ExprObject *else_expr;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &if_expr,
                          ExprType, &then_expr,
                          ExprType, &else_expr)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Conditional>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(if_expr->parent.ptr),
            unique_downcast<Expr, ASTNode>(then_expr->parent.ptr),
            unique_downcast<Expr, ASTNode>(else_expr->parent.ptr));
    return 0;
}

///////////////////////// PIPE /////////////////////////

AST_NODE(Pipe, Expr)
AST_NODE_SUBINIT(Pipe) {
    DebugInfoObject *dbg;
    ExprObject *left, *right;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &left,
                          ExprType, &right)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Pipe>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(left->parent.ptr),
            unique_downcast<Expr, ASTNode>(right->parent.ptr));
    return 0;
}

///////////////////////// INSTANTIATE /////////////////////////

AST_NODE(Instantiate, Expr)
AST_NODE_SUBINIT(Instantiate) {
    DebugInfoObject *dbg;
    ExprObject *var;
    std::vector<std::unique_ptr<DimVarExpr>> instance_vals;
    if (!PyArg_ParseTuple(args, "O!O!O&",
                          DebugInfoType, &dbg,
                          ExprType, &var,
                          convert_dimvarexpr_list, (void *)&instance_vals)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Instantiate>(
            std::move(dbg->ptr),
            std::move(var->parent.ptr),
            std::move(instance_vals));
    return 0;
}

///////////////////////// REPEAT /////////////////////////

AST_NODE(Repeat, Expr)
AST_NODE_SUBINIT(Repeat) {
    DebugInfoObject *dbg;
    ExprObject *body;
    DimVarExprObject *ub;
    const char *loopvar;
    if (!PyArg_ParseTuple(args, "O!O!sO!",
                          DebugInfoType, &dbg,
                          ExprType, &body,
                          &loopvar,
                          DimVarExprType, &ub)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Repeat>(
            std::move(dbg->ptr),
            std::move(body->parent.ptr),
            loopvar,
            std::move(ub->ptr));
    return 0;
}

///////////////////////// LEFT PRED /////////////////////////

AST_NODE(Pred, Expr)
AST_NODE_SUBINIT(Pred) {
    DebugInfoObject *dbg;
    ExprObject *basis;
    ExprObject *body;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &basis,
                          ExprType, &body)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Pred>(
            std::move(dbg->ptr),
            std::move(basis->parent.ptr),
            std::move(body->parent.ptr));
    return 0;
}

///////////////////////// BIT UNARY OP /////////////////////////

AST_NODE(BitUnaryOp, Expr)
AST_NODE_SUBINIT(BitUnaryOp) {
    DebugInfoObject *dbg;
    BitOp op;
    ExprObject *operand;
    if (!PyArg_ParseTuple(args, "O!iO!",
                          DebugInfoType, &dbg,
                          &op,
                          ExprType, &operand)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BitUnaryOp>(
            std::move(dbg->ptr),
            op,
            unique_downcast<Expr, ASTNode>(operand->parent.ptr));
    return 0;
}

///////////////////////// BIT BINARY OP /////////////////////////

AST_NODE(BitBinaryOp, Expr)
AST_NODE_SUBINIT(BitBinaryOp) {
    DebugInfoObject *dbg;
    BitOp op;
    ExprObject *left, *right;
    if (!PyArg_ParseTuple(args, "O!iO!O!",
                          DebugInfoType, &dbg,
                          &op,
                          ExprType, &left,
                          ExprType, &right)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BitBinaryOp>(
            std::move(dbg->ptr),
            op,
            unique_downcast<Expr, ASTNode>(left->parent.ptr),
            unique_downcast<Expr, ASTNode>(right->parent.ptr));
    return 0;
}

///////////////////////// BIT REDUCE OP /////////////////////////

AST_NODE(BitReduceOp, Expr)
AST_NODE_SUBINIT(BitReduceOp) {
    DebugInfoObject *dbg;
    BitOp op;
    ExprObject *operand;
    if (!PyArg_ParseTuple(args, "O!iO!",
                          DebugInfoType, &dbg,
                          &op,
                          ExprType, &operand)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BitReduceOp>(
            std::move(dbg->ptr),
            op,
            unique_downcast<Expr, ASTNode>(operand->parent.ptr));
    return 0;
}

///////////////////////// BIT CONCAT /////////////////////////

AST_NODE(BitConcat, Expr)
AST_NODE_SUBINIT(BitConcat) {
    DebugInfoObject *dbg;
    ExprObject *left, *right;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &left,
                          ExprType, &right)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BitConcat>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(left->parent.ptr),
            unique_downcast<Expr, ASTNode>(right->parent.ptr));
    return 0;
}

///////////////////////// BIT REPEAT /////////////////////////

AST_NODE(BitRepeat, Expr)
AST_NODE_SUBINIT(BitRepeat) {
    DebugInfoObject *dbg;
    ExprObject *bits;
    DimVarExprObject *amt;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &bits,
                          DimVarExprType, &amt)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BitRepeat>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(bits->parent.ptr),
            std::move(amt->ptr));
    return 0;
}

///////////////////////// MODULAR MULTIPLICATION /////////////////////////

AST_NODE(ModMulOp, Expr)
AST_NODE_SUBINIT(ModMulOp) {
    DebugInfoObject *dbg;
    DimVarExprObject *x, *j, *modN;
    ExprObject *y;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
                          DebugInfoType, &dbg,
                          DimVarExprType, &x,
                          DimVarExprType, &j,
                          ExprType, &y,
                          DimVarExprType, &modN)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<ModMulOp>(
            std::move(dbg->ptr),
            std::move(x->ptr),
            std::move(j->ptr),
            std::move(y->parent.ptr),
            std::move(modN->ptr));
    return 0;
}

///////////////////////// CLASSICAL BIT LITERAL /////////////////////////

AST_NODE(BitLiteral, Expr)
AST_NODE_SUBINIT(BitLiteral) {
    DebugInfoObject *dbg;
    DimVarExprObject *val, *n_bits;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          DebugInfoType, &dbg,
                          DimVarExprType, &val,
                          DimVarExprType, &n_bits)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<BitLiteral>(
            std::move(dbg->ptr),
            std::move(val->ptr),
            std::move(n_bits->ptr));
    return 0;
}

///////////////////////// SLICING /////////////////////////

AST_NODE(Slice, Expr)
AST_NODE_SUBINIT(Slice) {
    DebugInfoObject *dbg;
    ExprObject *val;
    PyObject *lower_obj, *upper_obj;
    if (!PyArg_ParseTuple(args, "O!O!OO",
                          DebugInfoType, &dbg,
                          ExprType, &val,
                          &lower_obj,
                          &upper_obj)) {
        return -1;
    }

    std::unique_ptr<DimVarExpr> lower, upper;
    if (Py_None == lower_obj) {
        lower = nullptr;
    } else if (Py_TYPE(lower_obj) == (PyTypeObject *)DimVarExprType) {
        DimVarExprObject *lower_wrapper = (DimVarExprObject *)lower_obj;
        lower = std::move(lower_wrapper->ptr);
    } else {
        PyErr_SetString(PyExc_ValueError, "Expected DimVarExpr as lower");
        return -1;
    }
    if (Py_None == upper_obj) {
        upper = nullptr;
    } else if (Py_TYPE(upper_obj) == (PyTypeObject *)DimVarExprType) {
        DimVarExprObject *upper_wrapper = (DimVarExprObject *)upper_obj;
        upper = std::move(upper_wrapper->ptr);
    } else {
        PyErr_SetString(PyExc_ValueError, "Expected DimVarExpr as upper");
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Slice>(
            std::move(dbg->ptr),
            std::move(val->parent.ptr),
            std::move(lower),
            std::move(upper));
    return 0;
}

///////////////////////// STMT /////////////////////////

AST_NODE(Stmt, ASTNode)
AST_NODE_SUBINIT(Stmt) { return 0; }

///////////////////////// ASSIGNMENT /////////////////////////

AST_NODE(Assign, Stmt)
AST_NODE_SUBINIT(Assign) {
    DebugInfoObject *dbg;
    const char *name;
    ExprObject *value;
    if (!PyArg_ParseTuple(args, "O!sO!",
                          DebugInfoType, &dbg,
                          &name,
                          ExprType, &value)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Assign>(
            std::move(dbg->ptr),
            name,
            unique_downcast<Expr, ASTNode>(value->parent.ptr));
    return 0;
}

///////////////////////// DESTRUCTURING /////////////////////////

AST_NODE(DestructAssign, Stmt)
AST_NODE_SUBINIT(DestructAssign) {
    DebugInfoObject *dbg;
    std::vector<std::string> target_names;
    ExprObject *value;
    if (!PyArg_ParseTuple(args, "O!O&O!",
                          DebugInfoType, &dbg,
                          convert_string_list, (void *)&target_names,
                          ExprType, &value)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<DestructAssign>(
            std::move(dbg->ptr),
            target_names,
            unique_downcast<Expr, ASTNode>(value->parent.ptr));
    return 0;
}

///////////////////////// RETURN /////////////////////////

AST_NODE(Return, Stmt)
AST_NODE_SUBINIT(Return) {
    DebugInfoObject *dbg;
    ExprObject *value;
    if (!PyArg_ParseTuple(args, "O!O!",
                          DebugInfoType, &dbg,
                          ExprType, &value)) {
        return -1;
    }

    self->parent.parent.ptr = std::make_unique<Return>(
            std::move(dbg->ptr),
            unique_downcast<Expr, ASTNode>(value->parent.ptr));
    return 0;
}

///////////////////////// KERNEL /////////////////////////

AST_NODE_BOILERPLATE(Kernel, ASTNode)
AST_NODE_SUBINIT(Kernel) {
    // Special case: building this from a Kernel pointer. We'll come back and
    // set the kernel pointer later
    if (!PyTuple_Size(args)) {
        return 0;
    // Special case: a copy
    } else if (PyTuple_Size(args) == 1) {
        KernelObject *other_kernel;
        if (!PyArg_ParseTuple(args, "O!", Py_TYPE(self), &other_kernel)) {
            return -1;
        }
        self->parent.ptr = std::move(other_kernel->parent.ptr->copy());
        return 0;
    }

    ASTKind ast_kind;
    const char *name;
    TypeObject *type;
    std::vector<std::string> capture_names;
    std::vector<std::unique_ptr<Type>> capture_types;
    std::vector<size_t> capture_freevars;
    std::vector<std::string> arg_names;
    std::vector<std::string> dimvars;
    std::vector<std::unique_ptr<ASTNode>> body;
    DebugInfoObject *dbg;

    if (!PyArg_ParseTuple(args, "O!isO!O&O&O&O&O&O&",
                          DebugInfoType, &dbg,
                          &ast_kind,
                          &name,
                          TypeType, &type,
                          convert_string_list, (void *)&capture_names,
                          convert_type_list, (void *)&capture_types,
                          convert_size_t_list, (void *)&capture_freevars,
                          convert_string_list, (void *)&arg_names,
                          convert_string_list, (void *)&dimvars,
                          convert_node_list, (void *)&body)) {
        return -1;
    }

    #define KERNEL_ARGS \
        std::move(dbg->ptr), \
        name, \
        std::move(type->ptr), \
        capture_names, \
        std::move(capture_types), \
        capture_freevars, \
        arg_names, \
        dimvars, \
        std::move(body)

    switch (ast_kind) {
    case AST_QPU:
        self->parent.ptr = std::make_unique<QpuKernel>(KERNEL_ARGS);
        break;
    case AST_CLASSICAL:
        self->parent.ptr = std::make_unique<ClassicalKernel>(KERNEL_ARGS);
        break;
    default:
        PyErr_Format(PyExc_ValueError, "Unknown AST Type %d", ast_kind);
        return -1;
    }
    return 0;
}

static PyObject *Kernel_infer_dimvars_from_captures(KernelObject *self,
                                                     PyObject *args) {
    std::vector<std::unique_ptr<HybridObj>> captures;
    if (!PyArg_ParseTuple(args, "O&",
                          convert_owned_hybrid_list, (void *)&captures)) {
        return NULL;
    }

    try {
        Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
        kernel.inferDimvarsFromCaptures(std::move(captures));
    } catch (TypeException &err) {
        qwerty_programmer_error(QwertyTypeError, err.message, *err.dbg);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *Kernel_register_explicit_dimvars(KernelObject *self, PyObject *args) {
    std::vector<std::optional<DimVarValue>> values;
    if (!PyArg_ParseTuple(args, "O&",
                          convert_dimvar_value_list, (void *)&values)) {
        return NULL;
    }

    try {
        Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
        kernel.registerExplicitDimvars(values);
    } catch (TypeException &err) {
        qwerty_programmer_error(QwertyTypeError, err.message, *err.dbg);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *Kernel_typecheck(KernelObject *self, PyObject *args) {
    (void)args;

    try {
        Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
        kernel.typeCheck();
    } catch (TypeException &err) {
        qwerty_programmer_error(QwertyTypeError, err.message, *err.dbg);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *Kernel_needs_recompile(KernelObject *self, PyObject *args) {
    std::vector<HybridObj *> captures;
    if (!PyArg_ParseTuple(args, "O&",
                          convert_borrowed_hybrid_list, (void *)&captures)) {
        return NULL;
    }

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    return PyBool_FromLong(kernel.needsRecompile(captures));
}

static PyObject *Kernel_compile(KernelObject *self, PyObject *args) {
    MlirHandleObject *handle;
    if (!PyArg_ParseTuple(args, "O!",
                          MlirHandleType, &handle)) {
        return NULL;
    }

    try {
        Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
        kernel.compile(*handle->ptr);
    } catch (CompileException &err) {
        qwerty_programmer_error(QwertyCompileError, err.message, *err.dbg);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *Kernel_call(KernelObject *self, PyObject *args) {
    MlirHandleObject *handle;
    const char *acc_ptr;
    unsigned long n_shots;
    if (!PyArg_ParseTuple(args, "O!zk",
                          MlirHandleType, &handle,
                          &acc_ptr,
                          &n_shots)) {
        return NULL;
    }

    std::string acc(acc_ptr? acc_ptr : "");

    std::unique_ptr<KernelResult> result;
    try {
        Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
        result = std::move(kernel.call(*handle->ptr, acc, n_shots));
    } catch (JITException &err) {
        PyErr_SetString(QwertyJITError, err.message.c_str());
        return NULL;
    }

    if (!result) {
        Py_RETURN_NONE;
    } else {
        PyObject *histo;
        if (!(histo = PyDict_New())) {
            return NULL;
        }
        for (const auto &[bits, count] : *result) {
            BitsObject *bits_obj = (BitsObject *)PyObject_CallNoArgs(BitsType);
            bits_obj->parent.ptr = std::make_unique<Bits>(bits);
            PyObject *count_obj;
            if (!(count_obj = PyLong_FromSize_t(count))) {
                Py_DECREF(histo);
                Py_DECREF(bits_obj);
                return NULL;
            }
            if (PyDict_SetItem(histo, (PyObject*)bits_obj, count_obj) < 0) {
                Py_DECREF(histo);
                Py_DECREF(bits_obj);
                Py_DECREF(count_obj);
                return NULL;
            }
        }
        return histo;
    }
}

static PyObject *Kernel_erase(KernelObject *self, PyObject *args) {
    (void)args;

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    kernel.erase();

    Py_RETURN_NONE;
}

static PyObject *Kernel_dump(KernelObject *self, PyObject *args) {
    (void)args;

    GraphvizVisitor visitor;
    self->parent.ptr->walk(visitor);
    return PyUnicode_FromString(visitor.str().c_str());
}

static PyObject *Kernel_get_name(KernelObject *self, PyObject *args) {
    (void)args;

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    return PyUnicode_FromString(kernel.name.c_str());
}

static PyObject *Kernel_get_dimvars(KernelObject *self, PyObject *args) {
    (void)args;

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    std::vector<DimVar> &dimvars = kernel.dimvars;
    PyObject *list = PyList_New(dimvars.size());
    if (!list) {
        return NULL;
    }
    for (ssize_t i = 0; i < (ssize_t)dimvars.size(); i++) {
        PyObject *str = PyUnicode_FromString(dimvars[i].c_str());
        if (!str || PyList_SetItem(list, i, str) < 0) {
            Py_XDECREF(str);
            Py_DECREF(list);
            return NULL;
        }
    }
    return list;
}

static PyObject *Kernel_get_explicit_dimvar_names(KernelObject *self, PyObject *args) {
    (void)args;

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    std::vector<DimVar> &explicit_dimvars = kernel.explicit_dimvars;
    PyObject *list = PyList_New(explicit_dimvars.size());
    if (!list) {
        return NULL;
    }
    for (ssize_t i = 0; i < (ssize_t)explicit_dimvars.size(); i++) {
        PyObject *str = PyUnicode_FromString(explicit_dimvars[i].c_str());
        if (!str || PyList_SetItem(list, i, str) < 0) {
            Py_XDECREF(str);
            Py_DECREF(list);
            return NULL;
        }
    }
    return list;
}

static PyObject *Kernel_needs_explicit_dimvars(KernelObject *self, PyObject *args) {
    (void)args;

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    return PyBool_FromLong(kernel.needs_explicit_dimvars);
}

static PyObject *Kernel_missing_dimvars(KernelObject *self, PyObject *args) {
    (void)args;

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    std::vector<DimVar> &missing_dimvars = kernel.missing_dimvars;
    PyObject *list = PyList_New(missing_dimvars.size());
    if (!list) {
        return NULL;
    }
    for (ssize_t i = 0; i < (ssize_t)missing_dimvars.size(); i++) {
        PyObject *str = PyUnicode_FromString(missing_dimvars[i].c_str());
        if (!str || PyList_SetItem(list, i, str) < 0) {
            Py_XDECREF(str);
            Py_DECREF(list);
            return NULL;
        }
    }
    return list;
}

static PyObject *Kernel_set_unique_gen_id(KernelObject *self, PyObject *args) {
    const char *unique_gen_id;
    if (!PyArg_ParseTuple(args, "s", &unique_gen_id)) {
        return NULL;
    }

    Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
    kernel.unique_gen_id = unique_gen_id;
    Py_RETURN_NONE;
}

// Not a general ASTNode method because not every ASTNode implements a copy
// constructor like Kernel does
static PyObject *Kernel_copy(KernelObject *self, PyObject *args) {
    (void)args;
    KernelObject *kernelCopy = (KernelObject *)PyObject_CallOneArg((PyObject *)Py_TYPE(self), (PyObject *)self);
    return (PyObject *)kernelCopy;
}

static PyObject *Kernel_qasm(KernelObject *self, PyObject *args) {
    MlirHandleObject *handle;
    int print_locs;
    if (!PyArg_ParseTuple(args, "O!p",
                          MlirHandleType, &handle,
                          &print_locs)) {
        return NULL;
    }

    std::string qasm;
    try {
        Kernel &kernel = static_cast<Kernel &>(*self->parent.ptr);
        qasm = kernel.qasm(*handle->ptr, print_locs);
    } catch (JITException &err) {
        PyErr_SetString(QwertyJITError, err.message.c_str());
        return NULL;
    }

    return PyUnicode_FromString(qasm.c_str());
}

AST_NODE_METHODS(Kernel) {
    {"infer_dimvars_from_captures", (PyCFunction)Kernel_infer_dimvars_from_captures, METH_VARARGS, "Attempt to infer type variables"},
    {"register_explicit_dimvars", (PyCFunction)Kernel_register_explicit_dimvars, METH_VARARGS, "Register explicit dimvars, validating the right number"},
    {"typecheck", (PyCFunction)Kernel_typecheck, METH_NOARGS, "Recursively typecheck a Kernel AST"},
    {"needs_recompile", (PyCFunction)Kernel_needs_recompile, METH_VARARGS, "Checks if captures provided match existing compiled version"},
    {"compile", (PyCFunction)Kernel_compile, METH_VARARGS, "Lower AST to MLIR"},
    {"call", (PyCFunction)Kernel_call, METH_VARARGS, "JIT if necessary and simulate Kernel"},
    {"erase", (PyCFunction)Kernel_erase, METH_NOARGS, "Remove FuncOp. Makes sense to do when the handle gets GC'd by Python"},
    {"dump", (PyCFunction)Kernel_dump, METH_NOARGS, "Return AST represented as a string"},
    {"get_name", (PyCFunction)Kernel_get_name, METH_NOARGS, "Return the name of this function as a string"},
    {"get_dimvars", (PyCFunction)Kernel_get_dimvars, METH_NOARGS, "Return the dimvars for this function"},
    {"get_explicit_dimvar_names", (PyCFunction)Kernel_get_explicit_dimvar_names, METH_NOARGS, "Return the names of explicit dimvars previously passed with mykernel[A,B,...Z]"},
    {"needs_explicit_dimvars", (PyCFunction)Kernel_needs_explicit_dimvars, METH_NOARGS, "Return whether dimvars should be explicitly passed when invoking this kernel"},
    {"missing_dimvars", (PyCFunction)Kernel_missing_dimvars, METH_NOARGS, "Return the dimvars that are missing if this kernel needs explicit dimvars"},
    {"set_unique_gen_id", (PyCFunction)Kernel_set_unique_gen_id, METH_VARARGS, "Set unique identifier for this generation of this kernel"},
    {"copy", (PyCFunction)Kernel_copy, METH_NOARGS, "Return copy of myself"},
    {"qasm", (PyCFunction)Kernel_qasm, METH_VARARGS, "Return OpenQASM 3 string of this kernel"},
    {NULL}
};

AST_NODE_TYPE_WITH_METHODS(Kernel, ASTNode)

static int convert_borrowed_hybrid_list(PyObject *hybrids_list, void *result_vector) {
    std::vector<HybridObj *> *output_vector =
            (std::vector<HybridObj *> *)result_vector;
    PyObject *iter = NULL;
    PyObject *hybrid_obj = NULL;
    if (!(iter = PyObject_GetIter(hybrids_list))) {
        goto failure;
    }
    while ((hybrid_obj = PyIter_Next(iter))) {
        int ret;
        HybridObj *hybrid;
        if ((ret = PyObject_IsInstance(hybrid_obj, HybridObjType)) == -1) {
            goto failure;
        } else if (ret) {
            HybridObjObject *obj = (HybridObjObject *)hybrid_obj;
            hybrid = obj->ptr.get();
        // TODO: get rid of this manual hack to check for Kernel. this is
        //       caused by it being inconvenient to use multiple inheritance in
        //       the Python C API
        } else if (!ret && Py_IS_TYPE(hybrid_obj, (PyTypeObject *)KernelType)) {
            KernelObject *obj = (KernelObject *)hybrid_obj;
            hybrid = static_cast<Kernel *>(obj->parent.ptr.get());
        } else {
            PyErr_SetString(PyExc_ValueError, "Need to pass list of HybridObjs");
            goto failure;
        }
        output_vector->push_back(hybrid);
        Py_XDECREF(hybrid_obj);
        hybrid_obj = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(hybrid_obj);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

static int convert_owned_hybrid_list(PyObject *hybrids_list, void *result_vector) {
    std::vector<std::unique_ptr<HybridObj>> *output_vector =
            (std::vector<std::unique_ptr<HybridObj>> *)result_vector;
    PyObject *iter = NULL;
    PyObject *hybrid_obj = NULL;
    if (!(iter = PyObject_GetIter(hybrids_list))) {
        goto failure;
    }
    while ((hybrid_obj = PyIter_Next(iter))) {
        int ret;
        std::unique_ptr<HybridObj> hybrid;
        if ((ret = PyObject_IsInstance(hybrid_obj, HybridObjType)) == -1) {
            goto failure;
        } else if (ret) {
            HybridObjObject *obj = (HybridObjObject *)hybrid_obj;
            hybrid = std::move(obj->ptr);
        // TODO: get rid of this manual hack to check for Kernel. this is
        //       caused by it being inconvenient to use multiple inheritance in
        //       the Python C API
        } else if (!ret && Py_IS_TYPE(hybrid_obj, (PyTypeObject *)KernelType)) {
            KernelObject *obj = (KernelObject *)hybrid_obj;
            hybrid = unique_downcast<Kernel, ASTNode>(obj->parent.ptr);
        } else {
            PyErr_SetString(PyExc_ValueError, "Need to pass list of HybridObjs");
            goto failure;
        }
        output_vector->push_back(std::move(hybrid));
        Py_XDECREF(hybrid_obj);
        hybrid_obj = NULL;
    }

    failure:
    Py_XDECREF(iter);
    Py_XDECREF(hybrid_obj);

    if (PyErr_Occurred()) {
        return 0;
    } else {
        return 1;
    }
}

///////////////////////// MODULE ITSELF /////////////////////////

static PyObject *qwerty_harness_set_debug(PyObject *self, PyObject *args) {
    (void)self;
    int debug;
    if (!PyArg_ParseTuple(args, "p", &debug)) {
        return NULL;
    }
    qwerty_debug = debug;
    Py_RETURN_NONE;
}

static PyObject *qwerty_harness_embedding_kind_name(PyObject *self, PyObject *args) {
    (void)self;
    EmbeddingKind kind;
    if (!PyArg_ParseTuple(args, "i", &kind)) {
        return NULL;
    }
    return PyUnicode_FromString(embedding_kind_name(kind).c_str());
}

static PyObject *qwerty_harness_embedding_kind_has_operand(PyObject *self, PyObject *args) {
    (void)self;
    EmbeddingKind kind;
    if (!PyArg_ParseTuple(args, "i", &kind)) {
        return NULL;
    }
    return PyBool_FromLong(embedding_kind_has_operand(kind));
}

static PyMethodDef qwerty_harnessmethods[] = {
    {"set_debug", qwerty_harness_set_debug, METH_VARARGS, "Enable or disable verbose logging"},
    {"embedding_kind_name", qwerty_harness_embedding_kind_name, METH_VARARGS, "Get string name of EmbeddingKind enum value"},
    {"embedding_kind_has_operand", qwerty_harness_embedding_kind_has_operand, METH_VARARGS, "Return whether an embedding takes an argument"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef qwerty_harnessmodule = {
    PyModuleDef_HEAD_INIT,
    "_qwerty_harness",
    NULL, // No module documentation
    -1, // No state
    qwerty_harnessmethods
};

// The definition of PyMODINIT_FUNC includes `extern "C"'. Pretty cool
PyMODINIT_FUNC PyInit__qwerty_harness(void) {
    PyObject *m = PyModule_Create(&qwerty_harnessmodule);
    if (!m) {
        return NULL;
    }

    // Types (includes heap allocation)
    AST_NODE_RUNTIME_SETUP(DimVarExpr);
    AST_NODE_RUNTIME_SETUP(DebugInfo);
    AST_NODE_RUNTIME_SETUP(Type);
    AST_NODE_RUNTIME_SETUP(ASTNode);
    AST_NODE_RUNTIME_SETUP(Expr);
    AST_NODE_RUNTIME_SETUP(Variable);
    AST_NODE_RUNTIME_SETUP(Adjoint);
    AST_NODE_RUNTIME_SETUP(Prepare);
    AST_NODE_RUNTIME_SETUP(LiftBits);
    AST_NODE_RUNTIME_SETUP(EmbedClassical);
    AST_NODE_RUNTIME_SETUP(BiTensor);
    AST_NODE_RUNTIME_SETUP(BroadcastTensor);
    AST_NODE_RUNTIME_SETUP(QubitLiteral);
    AST_NODE_RUNTIME_SETUP(Phase);
    AST_NODE_RUNTIME_SETUP(FloatLiteral);
    AST_NODE_RUNTIME_SETUP(FloatNeg);
    AST_NODE_RUNTIME_SETUP(FloatBinaryOp);
    AST_NODE_RUNTIME_SETUP(FloatDimVarExpr);
    AST_NODE_RUNTIME_SETUP(TupleLiteral);
    AST_NODE_RUNTIME_SETUP(BuiltinBasis);
    AST_NODE_RUNTIME_SETUP(Identity);
    AST_NODE_RUNTIME_SETUP(BasisTranslation);
    AST_NODE_RUNTIME_SETUP(Discard);
    AST_NODE_RUNTIME_SETUP(Measure);
    AST_NODE_RUNTIME_SETUP(Project);
    AST_NODE_RUNTIME_SETUP(Flip);
    AST_NODE_RUNTIME_SETUP(Rotate);
    AST_NODE_RUNTIME_SETUP(BasisLiteral);
    AST_NODE_RUNTIME_SETUP(Conditional);
    AST_NODE_RUNTIME_SETUP(Pipe);
    AST_NODE_RUNTIME_SETUP(Instantiate);
    AST_NODE_RUNTIME_SETUP(Repeat);
    AST_NODE_RUNTIME_SETUP(Pred);
    AST_NODE_RUNTIME_SETUP(BitUnaryOp);
    AST_NODE_RUNTIME_SETUP(BitBinaryOp);
    AST_NODE_RUNTIME_SETUP(BitReduceOp);
    AST_NODE_RUNTIME_SETUP(BitConcat);
    AST_NODE_RUNTIME_SETUP(BitRepeat);
    AST_NODE_RUNTIME_SETUP(ModMulOp);
    AST_NODE_RUNTIME_SETUP(BitLiteral);
    AST_NODE_RUNTIME_SETUP(Slice);
    AST_NODE_RUNTIME_SETUP(Stmt);
    AST_NODE_RUNTIME_SETUP(Assign);
    AST_NODE_RUNTIME_SETUP(DestructAssign);
    AST_NODE_RUNTIME_SETUP(Return);
    AST_NODE_RUNTIME_SETUP(Kernel);
    AST_NODE_RUNTIME_SETUP(HybridObj);
    AST_NODE_RUNTIME_SETUP(Bits);
    AST_NODE_RUNTIME_SETUP(Integer);
    AST_NODE_RUNTIME_SETUP(Angle);
    AST_NODE_RUNTIME_SETUP(Tuple);
    AST_NODE_RUNTIME_SETUP(MlirHandle);

    // Non-types
    PyObject *prog_err_fields = NULL;
    if (PyModule_AddIntMacro(m, PLUS) < 0
        || PyModule_AddIntMacro(m, MINUS) < 0
        || PyModule_AddIntMacro(m, X) < 0
        || PyModule_AddIntMacro(m, Y) < 0
        || PyModule_AddIntMacro(m, Z) < 0
        || PyModule_AddIntMacro(m, FOURIER) < 0
        || PyModule_AddIntMacro(m, EMBED_XOR) < 0
        || PyModule_AddIntMacro(m, EMBED_SIGN) < 0
        || PyModule_AddIntMacro(m, EMBED_INPLACE) < 0
        || PyModule_AddIntMacro(m, BIT_AND) < 0
        || PyModule_AddIntMacro(m, BIT_OR) < 0
        || PyModule_AddIntMacro(m, BIT_XOR) < 0
        || PyModule_AddIntMacro(m, BIT_NOT) < 0
        || PyModule_AddIntMacro(m, BIT_ROTL) < 0
        || PyModule_AddIntMacro(m, BIT_ROTR) < 0
        || PyModule_AddIntMacro(m, FLOAT_DIV) < 0
        || PyModule_AddIntMacro(m, FLOAT_POW) < 0
        || PyModule_AddIntMacro(m, FLOAT_MUL) < 0
        || PyModule_AddIntMacro(m, AST_QPU) < 0
        || PyModule_AddIntMacro(m, AST_CLASSICAL) < 0
        || !(prog_err_fields = PyDict_New())
        || PyDict_SetItemString(prog_err_fields, "dbg", Py_None) < 0
        || !(QwertyProgrammerError = PyErr_NewException("qwerty.QwertyProgrammerError", NULL, prog_err_fields))
        || PyModule_AddObjectRef(m, "QwertyProgrammerError", QwertyProgrammerError) < 0
        || !(QwertyTypeError = PyErr_NewException("qwerty.QwertyTypeError", QwertyProgrammerError, NULL))
        || PyModule_AddObjectRef(m, "QwertyTypeError", QwertyTypeError) < 0
        || !(QwertyCompileError = PyErr_NewException("qwerty.QwertyCompileError", QwertyProgrammerError, NULL))
        || PyModule_AddObjectRef(m, "QwertyCompileError", QwertyCompileError) < 0
        // Not a QwertyProgrammerError because it doesn't have a frame
        || !(QwertyJITError = PyErr_NewException("qwerty.QwertyJITError", NULL, NULL))
        || PyModule_AddObjectRef(m, "QwertyJITError", QwertyJITError) < 0) {

        // TODO: We may need more decrefs here, but I prefer leaking a
        //       little memory in a rare catastrophic failure than adding
        //       complicated control flow to decrement a bunch of refs
        Py_XDECREF(prog_err_fields);
        Py_XDECREF(QwertyTypeError);
        Py_XDECREF(QwertyCompileError);
        Py_XDECREF(QwertyJITError);
        return NULL;
    }

    return m;
}
