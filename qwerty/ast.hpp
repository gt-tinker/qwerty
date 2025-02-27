#ifndef AST_H
#define AST_H

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <set>
#include <map>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <optional>
#include <unordered_set>

#include "mockturtle/networks/xag.hpp"
#include "tweedledum/IR/Circuit.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include "Qwerty/IR/QwertyOps.h"
#include "Qwerty/IR/QwertyAttributes.h"
#include "Qwerty/IR/QwertyTypes.h"
#include "QCirc/IR/QCircTypes.h"

#include "defs.hpp"
#include "mlir_handle.hpp"
#include "ast_visitor.hpp"

// This file contains the declarations of all Qwerty AST nodes and types, as
// well as auxiliary types such as dimension variable expressions.

// Helper: cast a unique_ptr from a base class to a derived class
template<typename Derived, typename Base>
static std::unique_ptr<Derived> unique_downcast(std::unique_ptr<Base> &ptr)
{
    return std::unique_ptr<Derived>(dynamic_cast<Derived *>(ptr.release()));
}

// Copy lists of unique_ptr to things like AST nodes, types, etc that have
// copy() methods
template <typename T>
static inline std::vector<std::unique_ptr<T>> copy_vector_of_copyable(const std::vector<std::unique_ptr<T>> &old_vec) {
    std::vector<std::unique_ptr<T>> new_vec(old_vec.size());
    for (size_t i = 0; i < old_vec.size(); i++) {
        new_vec[i] = old_vec[i]->copy();
    }
    return new_vec;
}

// Similar to the above except for a vector of pairs whose right element is
// copy()able
template <typename S, typename T>
static inline std::vector<std::pair<S, std::unique_ptr<T>>> copy_vector_of_copyable_pair(
        const std::vector<std::pair<S, std::unique_ptr<T>>> &old_vec) {
    std::vector<std::pair<S, std::unique_ptr<T>>> new_vec;
    new_vec.reserve(old_vec.size());
    for (size_t i = 0; i < old_vec.size(); i++) {
        new_vec.emplace_back(old_vec[i].first, old_vec[i].second->copy());
    }
    return new_vec;
}

// Similar to copy_vector_of_copyable() except for comparison (i.e., the
// overload for `==' instead of `copy()')
template <typename T>
static inline bool compare_vectors(const std::vector<std::unique_ptr<T>> &lhs, const std::vector<std::unique_ptr<T>> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
        if (*lhs[i] != *rhs[i]) {
            return false;
        }
    }
    return true;
}

// Similar to compare_vectors() above except for a getHash() method instead of
// the operator overload for `=='.
template <typename T>
static inline bool compare_vectors_of_hashable(const std::vector<std::unique_ptr<T>> &lhs, const std::vector<std::unique_ptr<T>> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i]->getHash() != rhs[i]->getHash()) {
            return false;
        }
    }
    return true;
}

// Similar to compare_vectors() above except for pairs where the right element
// is a unique_ptr
template <typename S, typename T>
static inline bool compare_vectors_of_pairs(const std::vector<std::pair<S, std::unique_ptr<T>>> &lhs, const std::vector<std::pair<S, std::unique_ptr<T>>> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
        if (lhs[i].first != rhs[i].first
            || *lhs[i].second != *rhs[i].second) {
            return false;
        }
    }
    return true;
}

// Enums

// Kinds of embeddings of @classical functions
typedef enum EmbeddingKind {
    EMBED_XOR, // f.xor
    EMBED_SIGN, // f.sign
    EMBED_INPLACE, // f.inplace(f_inv)
} EmbeddingKind;

// Return the name of an embedding kind as a string. Useful for formatting
// error messages
static inline std::string embedding_kind_name(EmbeddingKind kind) {
    switch (kind) {
    case EMBED_XOR: return "xor";
    case EMBED_SIGN: return "sign";
    case EMBED_INPLACE: return "inplace";
    default: assert(0 && "Missing name of EmbeddingKind"); return "";
    }
}

// Return whether an embedding kind requires an operand. Currently this is just
// f.inplace(f_inv), but written this way for extensibility
static inline bool embedding_kind_has_operand(EmbeddingKind kind) {
    switch (kind) {
    case EMBED_XOR:
    case EMBED_SIGN:
    return false;
    case EMBED_INPLACE:
    return true; // Operand is the inverse function
    default: assert(0 && "Missing name of EmbeddingKind"); return false;
    }
}

// Kinds of Qwerty kernels
typedef enum {
    AST_QPU, // @qpu
    AST_CLASSICAL, // @classical
} ASTKind;

// Return the decorator name of a kind of Qwerty kernel. Useful for formatting
// error messages.
static inline std::string ast_kind_name(ASTKind kind) {
    switch (kind) {
    case AST_QPU: return "@qpu";
    case AST_CLASSICAL: return "@classical";
    default: assert(0 && "Missing decorator name of AST Kind"); return "";
    }
}

// [enum Eigenstate is defined in defs.hpp]

// Convert the AST Eigenstate enum to the Eigenstate enum in the Qwerty MLIR
// dialect. The ordering is the same, so we can thankfully just directly cast.
static inline qwerty::Eigenstate eigenstate_to_qwerty(Eigenstate e) {
    return (qwerty::Eigenstate)e;
}

// Return an eigenstate as a string. Useful for printing out ASTs
static inline std::string eigenstate_name(Eigenstate e) {
    switch (e) {
    case PLUS: return "+";
    case MINUS: return "-";
    default: assert(0 && "Missing name of Eigenstate"); return "";
    }
}

// [enum PrimitiveBasis is defined in defs.hpp]

// Convert the AST PrimitiveBasis enum to the PrimitiveBasis enum in the Qwerty
// MLIR dialect. The ordering is the same, so we can thankfully just directly
// cast.
static inline qwerty::PrimitiveBasis prim_basis_to_qwerty(PrimitiveBasis p) {
    return (qwerty::PrimitiveBasis)p;
}

// Return the Qwerty syntax for a given primitive basis. Useful for formatting
// error messages
static inline std::string prim_basis_name(PrimitiveBasis p) {
    switch (p) {
    case X: return "pm";
    case Y: return "ij";
    case Z: return "std";
    case FOURIER: return "fourier";
    default: assert(0 && "Missing basis name of PrimitiveBasis"); return "";
    }
}

// Kinds of bitwise operations in the @classical DSL
typedef enum BitOp {
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    BIT_NOT,
    BIT_ROTR,
    BIT_ROTL
} BitOp;

// Return a string representation of the kind of a bitwise operation. Useful
// for printing out a @classical AST
static inline std::string bit_op_name(BitOp bit_op) {
    switch (bit_op) {
    case BIT_AND: return "AND";
    case BIT_OR: return "OR";
    case BIT_XOR: return "XOR";
    case BIT_NOT: return "NOT";
    case BIT_ROTR: return "ROTR";
    case BIT_ROTL: return "ROTL";
    default: assert(0 && "Missing name of BitOp"); return "";
    }
}

// Return true if a bitwise operation requires like-sized operands. Useful for
// type checking
static inline bool bit_op_is_broadcast(BitOp bit_op) {
    switch (bit_op) {
    case BIT_AND:
    case BIT_OR:
    case BIT_XOR:
    case BIT_NOT:
    // Dimensions of bits need to match exactly
    return true;

    case BIT_ROTR:
    case BIT_ROTL:
    // Dimensions of bits do not need to match exactly
    return false;

    default: assert(0 && "Missing broadcast-ness of BitOp"); return false;
    }
}

// Kinds of operations on floats in the @qpu DSL
typedef enum FloatOp {
    FLOAT_DIV,
    FLOAT_POW,
    FLOAT_MUL,
    FLOAT_ADD,
    FLOAT_MOD
} FloatOp;

// Return a string representation of a float operation. Useful for printing out
// an AST
static inline std::string float_op_name(FloatOp float_op) {
    switch (float_op) {
    case FLOAT_DIV: return "DIV";
    case FLOAT_POW: return "POW";
    case FLOAT_MUL: return "MUL";
    case FLOAT_ADD: return "ADD";
    case FLOAT_MOD: return "MOD";
    default: assert(0 && "Missing name of FloatOp"); return "";
    }
}

// Abstract base class for a basis element in span checking. Used in type
// checking basis translations
struct Span {
    virtual ~Span() {}
    virtual size_t getNumQubits() const = 0;
    virtual std::unique_ptr<Span> copy() const = 0;
    // Return true if this span is equal to 𝓗_2⊗𝓗_2⊗...⊗𝓗_2, i.e., the span of
    // this basis element is the entire n-qubit space. (Here, n is the
    // dimension of this basis element, that is, `n = getNumQubits()')
    virtual bool fullySpans() const = 0;
    // Useful for debugging
    virtual void dump(std::ostream &os) const = 0;
};

std::ostream &operator<<(std::ostream &os, const Span &sp);

// A fully-spanning basis element such as std[N]
struct FullSpan : Span {
    size_t n_qubits;

    FullSpan(size_t n_qubits) : n_qubits(n_qubits) {}

    virtual size_t getNumQubits() const override { return n_qubits; }
    virtual std::unique_ptr<Span> copy() const override {
        return std::make_unique<FullSpan>(n_qubits);
    }
    virtual bool fullySpans() const override { return true; }
    virtual void dump(std::ostream &os) const override {
        os << "std[" << n_qubits << "]";
    }
};

// https://en.cppreference.com/w/cpp/utility/hash
template<>
struct std::hash<llvm::APInt>
{
    size_t operator()(const llvm::APInt &i) const noexcept {
        return (size_t)llvm::hash_value(i);
    }
};

// A basis literal that may or may not fully span
struct VeclistSpan : Span {
    PrimitiveBasis prim_basis;
    std::unordered_set<llvm::APInt> vecs;

    VeclistSpan(PrimitiveBasis prim_basis) : prim_basis(prim_basis) {}
    VeclistSpan(PrimitiveBasis prim_basis, const std::unordered_set<llvm::APInt> &vecs)
               : prim_basis(prim_basis), vecs(vecs) {}

    virtual size_t getNumQubits() const override {
        assert(!vecs.empty() && "VeclistSpan is empty, illegal");
        return (*vecs.begin()).getBitWidth();
    }

    virtual std::unique_ptr<Span> copy() const override {
        return std::make_unique<VeclistSpan>(prim_basis, vecs);
    }

    virtual bool fullySpans() const override {
        size_t n_qubits = getNumQubits();
        size_t two_to_the_n = 1ULL << n_qubits;
        return vecs.size() == two_to_the_n;
    }

    virtual void dump(std::ostream &os) const override;
};

// Represents the span of a basis in canon form (see Section 2.2 of the CGO
// paper), that is, it is a sequence of basis elements, specifically just the
// span of them.
struct SpanList {
    llvm::SmallVector<std::unique_ptr<Span>> list;

    void clear() { list.clear(); }
    // Add a new basis element to the end
    void append(std::unique_ptr<Span> sp) {
        list.push_back(std::move(sp));
    }
    // Concate another SpanList onto the end of this one
    void append(const SpanList &span) {
        for (const std::unique_ptr<Span> &sp : span.list) {
            list.push_back(sp->copy());
        }
    }
    // Do all basis elements fully span?
    bool fullySpans() const {
        assert(!list.empty() && "list of spans should not be empty");
        for (const std::unique_ptr<Span> &sp : list) {
            if (!sp->fullySpans()) {
                return false;
            }
        }
        return true;
    }

    friend bool operator==(const SpanList &left, const SpanList &right);
    friend bool operator!=(const SpanList &left, const SpanList &right) {
        return !(left == right);
    }
};

std::ostream &operator<<(std::ostream &os, const SpanList &span);

// Abstract base class used to represent captures. Each HybridObj has a hash
// (currently just a string) used to determine if a Qwerty kernel definition
// has been re-encountered with a different capture value, forcing a
// re-compile.
struct HybridObj {
    using Hash = std::string;

    virtual ~HybridObj() {}
    // Used for dimension variable inference
    virtual const Type &getType() const = 0;
    // Why not call this copy()? That was taken by ASTNode::copy() already, and
    // Kernel subclasses both HybridObj and ASTNode
    virtual std::unique_ptr<HybridObj> copyHybrid() const = 0;
    // Return the hash described above
    virtual Hash getHash() const = 0;

    // By default, none of the following methods do anything since they are currently
    // overridden only by Kernel.

    // Return true if and only if this capture still needs its own dimension
    // variables defined and can't be captured until then... unless dimension
    // variables are manually provided with capture[[1,2]] syntax.
    virtual bool needsExplicitDimvars() const { return false; }
    // Return the explicit dimension variables this capture is missing.
    virtual void getMissingDimvars(std::vector<DimVar> &missing_dimvars_out) const {}
    // If we were able to infer the dimension variables for this capture, then
    // evaluate all dimension variable expressions and run type checking
    virtual void evalExplicitDimvars(DimVarValues &vals, size_t n_freevars) {}
    // Instantiate this capture with the provided dimension variables (to
    // support the capture[[1,2]] syntax)
    virtual void instantiateWithExplicitDimvars(std::vector<DimVarValue> &values) {}
    // Compile this capture to an MLIR FuncOp if it hasn't been already. This
    // should call this method on its own captures too if needed. The
    // funcOp_private flag determines whether this capture could be used by
    // other functions or if we are the only user (e.g., if we instantiated
    // it ourselves).
    virtual void compileIfNeeded(MlirHandle &handle,
                                 std::string funcOp_name,
                                 bool funcOp_private) {}
    // Used by the caller (i.e., the capturer) to clean up any captures it
    // created that it doesn't need anymore. This is nice to avoid cluttering
    // the IR with unused symbols.
    virtual void eraseIfPrivate() {}
};

// Thrown during evaluation of dimension variable expressions when two sources
// of inference imply different concrete values for a dimension variable. We
// try to catch this and rethrow it as a TypeException so that
// _qwerty_harness.cpp can convert it into a more programmer-friendly error
// message.
struct DimVarInferenceConflict : public std::runtime_error {
    std::string message;

    DimVarInferenceConflict(std::string message)
                            : std::runtime_error(message),
                              message(message) {}
};

// Specific case of DimVarInferenceConflict thrown if you write e.g. cfunc[M,N]
// in the kernel signature but then provide a capture of type bit[5].
struct DimVarInferenceTypeMismatch : DimVarInferenceConflict {
    DimVarInferenceTypeMismatch(const Type &constType, const Type &genericType);
};

// Thrown when a dimension variable expression contains an undefined dimension
// variable. We try to catch and re-throw as a TypeException for programmers'
// sanity
struct MissingDimVarException : public std::runtime_error {
    DimVar missingDimvar;

    MissingDimVarException(DimVar missingDimvar)
                           : std::runtime_error("MissingDimVarException"),
                             missingDimvar(missingDimvar) {}
};

// Thrown when a dimension variable is evaluated to a negative number, which is
// not allowed. We also try to catch and re-throw as a TypeException for
// programmers' sanity
struct NegativeDimVarExprException : public std::runtime_error {
    std::vector<DimVar> offendingDimvars;

    NegativeDimVarExprException() : std::runtime_error("NegativeDimVarExprException") {}
    NegativeDimVarExprException(std::vector<DimVar> offendingDimvars)
                              : std::runtime_error("NegativeDimVarExprException"),
                                offendingDimvars(offendingDimvars) {}
};

// Thrown to represent a type mismatch in Qwerty code, usually by the type
// checker. _qwerty_harness.cpp catches this and rethrows it as a Python
// exception — see err.py for information on how it uses the DebugInfo in this
// exception to make the traceback point to the proper line of the programmer's
// code.
struct TypeException : public std::runtime_error {
    std::string message;
    std::unique_ptr<DebugInfo> dbg;

    TypeException(std::string message,
                  std::unique_ptr<DebugInfo> dbg)
                 : std::runtime_error(message),
                   message(message),
                   dbg(std::move(dbg)) {}
};

// Thrown deeper in the pipeline than TypeException (usually when lowering the
// AST to MLIR) but caught and re-thrown similarly. Typically indicates a
// bug/limitation in the compiler rather than a programmer mistake.
struct CompileException : public std::runtime_error {
    std::string message;
    std::unique_ptr<DebugInfo> dbg;

    CompileException(std::string message,
                     std::unique_ptr<DebugInfo> dbg)
                    : std::runtime_error(message),
                      message(message),
                      dbg(std::move(dbg)) {}
};

// [DimVar, DimVarValue, and DimVarValues typedefs are in defs.hpp]

// A dimension variable expression: an integer linear combination of dimension
// variables, plus a constant offset.
struct DimVarExpr {
    // Mapping of dimvar->factor
    // i.e., this dimvarexpr represents f1*dv1 + f2*dv2 + ... + fn+dvn + offset
    DimVarValues dimvars;
    DimVarValue offset;

    // For convenience: just pass 1 dimvar and 1 offset. To get a constant 3,
    // for example, pass dimvar="" and offset=3.
    DimVarExpr(DimVar dimvar, DimVarValue offset)
             : offset(offset) {
        if (!dimvar.empty()) {
            dimvars[dimvar] = 1;
        }
    }
    DimVarExpr(DimVarValue offset) : offset(offset) {}

    DimVarExpr(DimVarValues dimvars, DimVarValue offset)
             : dimvars(dimvars), offset(offset) {}

    // Return true if this expression consists only of an integer constant.
    // This should always be true after dimension variable expressions are
    // evaluated.
    bool isConstant() const { return dimvars.empty(); }

    std::unique_ptr<DimVarExpr> copy() const {
        return std::move(std::make_unique<DimVarExpr>(dimvars, offset));
    }

    // Infer as many dimension variables as we can and put the result in `out'.
    // For example, if constDimVarExpr=5 and *this is N+3, then we can conclude
    // N=2, so this method sets out["N"] = 2. Throws DimVarInferenceConflict if
    // there is a conflicting/different value already in `out'.
    void inferDimvars(const DimVarExpr &constDimVarExpr, DimVarValues &out) const;

    // Evaluate this expression with the provided dimension variable values
    // in-place. If permissive==false, then MissingDimVarException will be
    // thrown if *this has dimension variables that are not present in
    // `dimvar_values.' Regardless, will throw NegativeDimVarExprException if
    // the final offset calculated is negative — it doesn't make sense to have
    // an array of qubits of size -1!
    void eval(DimVarValues &dimvar_values, bool permissive);

    bool operator==(const DimVarExpr &other) const {
        return dimvars == other.dimvars && offset == other.offset;
    }

    bool operator!=(const DimVarExpr &other) const {
        return !(*this == other);
    }

    // The following overloads mutate this dimension variable expression
    // in-place.
    DimVarExpr &operator+=(const DimVarExpr &other);
    DimVarExpr &operator-=(const DimVarExpr &other);
    // At least one of the constant expressions operands must be constant for
    // this to succeed, since dimension variable expressions are linear
    // combinations of dimension variables, not general polynomials!
    DimVarExpr &operator*=(const DimVarExpr &other);
    std::string toString() const;
};

// Used to map an AST node back to a location in the original source code.
// Eventually converted to an mlir::Location
struct DebugInfo {
    // TODO: this is going to get duplicated like crazy
    std::string srcfile;
    unsigned int row;
    unsigned int col;
    // To avoid having to #include Python, this is a void*. But it is a
    // PyObject*, really. To what? The Python frame object at the time the AST
    // was parsed. This is necessary for concocting a good backtrace when type
    // checking etc fails. (See err.py)
    void *python_frame;

    DebugInfo(std::string srcfile, unsigned int row, unsigned int col,
              void *python_frame)
             : srcfile(srcfile), row(row), col(col),
               python_frame(python_frame) {
        python_incref(python_frame); // Python C API refcounting
    }

    ~DebugInfo() {
        python_decref(python_frame); // Python C API refcounting
    }

    bool operator==(const DebugInfo &dbg) const {
        return srcfile == dbg.srcfile
               && row == dbg.row
               && col == dbg.col
               && python_frame == dbg.python_frame;
    }
    bool operator!=(const DebugInfo &dbg) const {
        return !(*this == dbg);
    }

    // Defined in _qwerty_harness (don't want to call the Python API in
    // compiler code)
    static void python_incref(void *python_object);
    static void python_decref(void *python_object);

    std::unique_ptr<DebugInfo> copy() const {
        return std::move(std::make_unique<DebugInfo>(srcfile, row, col, python_frame));
    }

    mlir::Location toMlirLoc(MlirHandle &handle) const {
        return mlir::FileLineColLoc::get(&handle.context, srcfile, row, col);
    }
};

// Abstract base class for a type annotation for a Qwerty AST node
struct Type {
    virtual ~Type() {}
    // This shouldn't be called except in the implementation of operator<=. Use
    // that to check the subtype relation
    virtual bool isSubtypeOf(const Type &type) const = 0;
    virtual std::unique_ptr<Type> copy() const = 0;
    // Return true if and only if all dimension variable expressions contained
    // in this type are constant. Should always be true after evaluating
    // dimension variable expressions.
    virtual bool isConstant() const = 0;
    // Are adjacent registers/arrays merged together? Return true if so.
    // TODO: In future versions of Qwerty, taking the tensor product of
    //       registers/arrays should always merge them, making this someday
    //       obsolete (hopefully)
    virtual bool isCanonical() const = 0;
    // Useful for printing ASTs and error messages
    virtual std::string toString() const = 0;
    // Is a term of this type classically transmittable/serializable? For
    // example, a bit[N] is obviously, since you would just write down the N
    // bits it contains. A qubit would not be, however, since that is a fragile
    // precious quantum state. A function would always be since you could write
    // down the code for the function classically on a piece of paper.
    virtual bool isClassical() const = 0;
    // Can a @classical kernel take this in as an argument and return it as a
    // result? Return true if so.
    virtual bool isReversibleFriendly() const = 0;
    // Is this a linear type (e.g., a qubit)? Return true if so.
    virtual bool isLinear() const { return !isClassical(); }
    // Can an expression with this type be lowered to a MLIR Value? A Qubit
    // could be, for example, but a Basis could not
    virtual bool isMaterializable() const { return true; }
    // If this returns true, this type needs to be canonicalized ASAP, before
    // typechecking. Currently just BroadcastType
    virtual bool isFurled() const { return false; }
    // Return a new version of this type with all BroadcastTypes
    // unfurled/expanded to TupleTypes with the broadcasted element repeated.
    virtual std::unique_ptr<Type> unfurl() const { return std::move(copy()); }
    // Return a new version of myself with all adjacent registers/arrays in
    // tuples merged. See TODO above for isCanonicalize().
    virtual std::unique_ptr<Type> canonicalize() const = 0;
    // Collapse a TupleType with a repeated element back into a BroadcastType.
    // Useful for dimension variable inference.
    virtual std::unique_ptr<BroadcastType> collapseToBroadcast(const BroadcastType &genericType) const;
    // Calling this method is a humble request to call inferDimvars() on every
    // DimVarExpr contained in this type.
    virtual void doInferDimvars(const Type &constType, DimVarValues &out) const = 0;
    // Calling this method is a humble request to call eval() on every
    // DimVarExpr contained in this type. The `permissive' argument carries the
    // same meaning as in DimVarExpr::eval().
    virtual void evalDimVarExprs(DimVarValues &dimvar_values, bool permissive) = 0;
    // If this type is the type of a term t, then return the type of a term
    // t.xor (or t.sign or something else, based on `kind'). Throw a
    // TypeException if this doesn't make sense for this type.
    virtual std::unique_ptr<Type> asEmbedded(DebugInfo &dbg, EmbeddingKind kind) const {
        throw new TypeException("Cannot embed the type " + toString(), std::move(dbg.copy()));
    }
    // Convert this type to a sequence of MLIR types in the Qwerty dialect and
    // return them.
    // TODO: store this vector here and return a reference to avoid unnecessary
    //       copying.
    virtual llvm::SmallVector<mlir::Type> toMlirType(MlirHandle &handle) const = 0;
    // Return all child types of this one. Useful for traversing a type like a
    // tree.
    virtual std::vector<const Type *> children() const { return {}; };
    // Tensor product, effectively
    virtual std::unique_ptr<Type> operator+(const Type &type) const;

    // Subtype relation, aka <: (see Chapter 15 of TAPL)
    bool operator<=(const Type &type) const {
        const Type *lhs = this;
        const Type *rhs = &type;

        std::unique_ptr<Type> lhs_canon, rhs_canon;
        if (!lhs->isCanonical()) {
            lhs_canon = std::move(lhs->canonicalize());
            lhs = lhs_canon.get();
        }
        if (!rhs->isCanonical()) {
            rhs_canon = std::move(rhs->canonicalize());
            rhs = rhs_canon.get();
        }

        return typeid(*lhs) == typeid(*rhs) && lhs->isSubtypeOf(*rhs);
    }

    // Similar to inferDimvars() above but canonicalizes both types first
    void inferDimvars(const Type &constType, DimVarValues &out) const {
        // TODO: make some helper function to combine this with operator<= above
        const Type *lhs = this;
        const Type *rhs = &constType;

        std::unique_ptr<Type> lhs_canon, rhs_canon;
        if (!lhs->isCanonical()) {
            lhs_canon = std::move(lhs->canonicalize());
            lhs = lhs_canon.get();
        }
        if (!rhs->isCanonical()) {
            rhs_canon = std::move(rhs->canonicalize());
            rhs = rhs_canon.get();
        }

        lhs->doInferDimvars(*rhs, out);
    }

    bool operator==(const Type &type) const {
        return *this <= type && type <= *this;
    }

    bool operator!=(const Type &type) const {
        return !(*this == type);
    }

    // Recursively search for a different Type inside of this Type.
    // Confusingly, T is the C++ type of a Qwerty Type.
    template<typename T>
    bool contains() const {
        if (typeid(*this) == typeid(T)) {
            return true;
        }
        for (const Type *child : children()) {
            if (child->contains<T>()) {
                return true;
            }
        }
        return false;
    }

    // Returns a non-null pointer if this type can be collapsed to a BroadcastType
    // with element type T. For example, (angle, angle, angle) can be collapsed to
    // an angle[3], but (angle, angle, int) cannot.
    template<typename T>
    std::unique_ptr<BroadcastType> collapseToHomogeneousArray() const;
};

// Corresponds to e.g. angle[5] in Qwerty code. unfurl() will expand that to
// the Tuple type (angle, angle, angle, angle, angle), though. So this type is
// pretty short-lived and exists only for programmer convenience.
struct BroadcastType : Type {
    // In the example above, here's what the fields below would be:
    //     angle[5]
    //     \___/ ^
    //       |    \__ factor
    //       |
    //     elem_type
    std::unique_ptr<Type> elem_type;
    std::unique_ptr<DimVarExpr> factor;

    BroadcastType(std::unique_ptr<Type> elem_type,
                  std::unique_ptr<DimVarExpr> factor)
                 : elem_type(std::move(elem_type)),
                   factor(std::move(factor)) {}

    virtual bool isSubtypeOf(const Type &type) const override {
        const BroadcastType &broad = static_cast<const BroadcastType &>(type);
        return *elem_type <= *broad.elem_type && *factor == *broad.factor;
    }
    virtual std::unique_ptr<Type> copy() const override {
        return std::make_unique<BroadcastType>(std::move(elem_type->copy()),
                                               std::move(factor->copy()));
    }
    virtual bool isConstant() const override {
        return elem_type->isConstant() && factor->isConstant();
    }
    // We may not have enough information to canonicalize until unfurled
    virtual bool isCanonical() const override { return true; }
    // Needs to be unfurled by EvalDimVarExprs
    virtual bool isFurled() const override { return true; }
    virtual bool isClassical() const override {
        return elem_type->isClassical();
    }
    virtual bool isReversibleFriendly() const override {
        return elem_type->isReversibleFriendly();
    }
    virtual bool isMaterializable() const override {
        return elem_type->isMaterializable();
    }
    // See comment in isCanonical()
    virtual std::unique_ptr<Type> canonicalize() const override {
        return std::move(copy());
    }
    // Called by evaldimvarexprs because isFurled()==true. Basically gets rid of
    // this type
    virtual std::unique_ptr<Type> unfurl() const override;
    virtual void doInferDimvars(const Type &constType, DimVarValues &out) const override {
        std::unique_ptr<BroadcastType> constBroad = std::move(constType.collapseToBroadcast(*this));

        if (!(*constBroad->elem_type <= *elem_type)) {
            throw DimVarInferenceTypeMismatch(constType, *this);
        }
        factor->inferDimvars(*constBroad->factor, out);
    }
    virtual std::string toString() const override {
        return elem_type->toString() + "[" + factor->toString() + "]";
    }
    virtual void evalDimVarExprs(DimVarValues &dimvar_values, bool permissive) override {
        elem_type->evalDimVarExprs(dimvar_values, permissive);
        factor->eval(dimvar_values, permissive);
    }
    virtual llvm::SmallVector<mlir::Type> toMlirType(MlirHandle &handle) const override {
        return canonicalize()->toMlirType(handle);
    }
    virtual std::vector<const Type *> children() const override {
        return {elem_type.get()};
    };
};

// See the declaration of this method above for an explanation of what it is.
// This is mysteriously down here because it's a template so it can't be in a
// .cpp file, yet it calls the BroadcastType constructor, so it has to be after
// the BroadcastType constructor is declared directly above.
template<typename T>
std::unique_ptr<BroadcastType> Type::collapseToHomogeneousArray() const {
    std::unique_ptr<BroadcastType> furled;
    BroadcastType array(std::make_unique<T>(),
                        /*factor=*/nullptr);
    try {
        furled = collapseToBroadcast(array);
    } catch (DimVarInferenceConflict &exc) {
        return nullptr;
    }

    if (!(*furled->elem_type <= *array.elem_type)) {
        return nullptr;
    }

    return furled;
}

// The tensor product of n>1 types. Currently, Qwerty programmers do not write
// this type explicitly. Instead, when they write types on a function like
//     @qpu
//     def f(arg1: T1, arg2: T2, ..., argn: Tn) -> T':
//         # ...
// then the type of f is (T1, T2, ..., Tn) → T', where the part is parens is a
// TupleType.
// (TODO: Replace this design with Currying instead? Someday?)
//
// Tuples with adjacent array types, e.g., (Qubit[2], Qubit[3]) may be
// canonicalized by consoldating the adjacent array types, e.g., Qubit[5], by
// canonicalize().
struct TupleType : Type {
    std::vector<std::unique_ptr<Type>> types;

    TupleType() {} // Unit
    TupleType(std::vector<std::unique_ptr<Type>> types) : types(std::move(types)) {
        assert(types.size() != 1 && "A tuple of size 1 is redundant");
    }

    virtual bool isSubtypeOf(const Type &type) const override {
        const TupleType &tupleType = static_cast<const TupleType &>(type);
        if (tupleType.types.size() != types.size()) {
            return false;
        }
        for (size_t i = 0; i < types.size(); i++) {
            if (!(*types[i] <= *tupleType.types[i])) {
                return false;
            }
        }
        return true;
    }
    virtual std::unique_ptr<Type> copy() const override {
        return std::make_unique<TupleType>(std::move(copy_vector_of_copyable<Type>(types)));
    }
    virtual bool isConstant() const override {
        for (size_t i = 0; i < types.size(); i++) {
            if (!types[i]->isConstant()) {
                return false;
            }
        }
        return true;
    }
    virtual bool isCanonical() const override;
    virtual bool isFurled() const override {
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i]->isFurled()) {
                return true;
            }
        }
        return false;
    }
    virtual bool isClassical() const override {
        for (size_t i = 0; i < types.size(); i++) {
            if (!types[i]->isClassical()) {
                return false;
            }
        }
        return true;
    }
    virtual bool isReversibleFriendly() const override {
        for (size_t i = 0; i < types.size(); i++) {
            if (!types[i]->isReversibleFriendly()) {
                return false;
            }
        }
        return true;
    }
    virtual bool isMaterializable() const override {
        for (size_t i = 0; i < types.size(); i++) {
            if (!types[i]->isMaterializable()) {
                return false;
            }
        }
        return true;
    }
    virtual std::unique_ptr<Type> canonicalize() const override;
    virtual std::unique_ptr<Type> unfurl() const override;
    virtual std::unique_ptr<BroadcastType> collapseToBroadcast(const BroadcastType &genericType) const override {
        if (isUnit()) {
            return std::make_unique<BroadcastType>(std::move(genericType.elem_type->copy()),
                                                   std::make_unique<DimVarExpr>("", 0));
        }
        std::unique_ptr<BroadcastType> result = types[0]->collapseToBroadcast(genericType);
        for (size_t i = 1; i < types.size(); i++) {
            std::unique_ptr<BroadcastType> child_broad = types[i]->collapseToBroadcast(genericType);
            if (!(*child_broad->elem_type <= *genericType.elem_type)) {
                throw DimVarInferenceTypeMismatch(*this, genericType);
            }
            *result->factor += *child_broad->factor;
        }
        return std::move(result);
    }
    virtual void doInferDimvars(const Type &constType, DimVarValues &out) const override {
        const TupleType *constTupleType = dynamic_cast<const TupleType *>(&constType);
        if (constTupleType && constTupleType->types.size() == types.size()) {
            for (size_t i = 0; i < types.size(); i++) {
                types[i]->doInferDimvars(*constTupleType->types[i], out);
            }
        } else {
            throw DimVarInferenceTypeMismatch(constType, *this);
        }
    }
    virtual std::string toString() const override;
    virtual void evalDimVarExprs(DimVarValues &dimvar_values, bool permissive) override {
        for (size_t i = 0; i < types.size(); i++) {
            types[i]->evalDimVarExprs(dimvar_values, permissive);
        }
    }
    virtual llvm::SmallVector<mlir::Type> toMlirType(MlirHandle &handle) const override {
        llvm::SmallVector<mlir::Type> mlirTypes(types.size());
        for (size_t i = 0; i < types.size(); i++) {
            llvm::SmallVector<mlir::Type> range = types[i]->toMlirType(handle);
            // TODO: Is this check still needed? We do not allow nested tuples
            // TODO: Add a tuple MLIR type to fix this
            assert(range.size() == 1 && "I currently can't handle tuples inside of tuples");
            mlirTypes[i] = range[0];
        }
        return mlirTypes;
    }

    virtual std::vector<const Type *> children() const override {
        std::vector<const Type *> ret;
        for (size_t i = 0; i < types.size(); i++) {
            ret.push_back(types[i].get());
        }
        return ret;
    };

    bool isUnit() const { return types.empty(); }

    std::unique_ptr<Type> slice(size_t start_idx, size_t n_elem) const {
        assert(start_idx < types.size()
               && start_idx+n_elem <= types.size()
               && "slice indices out of range");

        if (n_elem == 1) {
            return std::move(types[start_idx]->copy());
        } else {
            std::vector<std::unique_ptr<Type>> sliced_types;
            sliced_types.reserve(n_elem);
            for (size_t i = 0; i < n_elem; i++) {
                sliced_types.push_back(std::move(types[start_idx + i]->copy()));
            }
            return std::make_unique<TupleType>(std::move(sliced_types));
        }
    }

    // Tensor product, effectively
    friend std::unique_ptr<Type> operator+(const TupleType &lhs, const Type &rhs) {
        std::vector<std::unique_ptr<Type>> new_types(lhs.types.size() + 1);
        for (size_t i = 0; i < lhs.types.size(); i++) {
            new_types[i] = std::move(lhs.types[i]->copy());
        }
        new_types[lhs.types.size()] = std::move(rhs.copy());
        return std::make_unique<TupleType>(std::move(new_types));
    }

    friend std::unique_ptr<Type> operator+(const Type &lhs, const TupleType &rhs) {
        std::vector<std::unique_ptr<Type>> new_types(rhs.types.size() + 1);
        for (size_t i = 0; i < rhs.types.size(); i++) {
            new_types[i+1] = std::move(rhs.types[i]->copy());
        }
        new_types[0] = std::move(lhs.copy());
        return std::make_unique<TupleType>(std::move(new_types));
    }

    friend std::unique_ptr<Type> operator+(const TupleType &lhs, const TupleType &rhs) {
        std::vector<std::unique_ptr<Type>> new_types(lhs.types.size() + rhs.types.size());
        for (size_t i = 0; i < lhs.types.size(); i++) {
            new_types[i] = std::move(lhs.types[i]->copy());
        }
        for (size_t i = 0; i < rhs.types.size(); i++) {
            new_types[i + lhs.types.size()] = std::move(rhs.types[i]->copy());
        }
        return std::make_unique<TupleType>(std::move(new_types));
    }
};


// Avoid repeating ourselves by defining a macro for the near-identical
// qubit[N], bit[N], and basis[N] types.
#define PSEUDO_ARR_TYPE(name, is_classical, is_rev_friendly, is_materializable, rev_impl, type_gen, extra_fields, extra_copy) \
    struct name ## Type : Type { \
        std::unique_ptr<DimVarExpr> dim; \
        \
        extra_fields \
        \
        name ## Type(std::unique_ptr<DimVarExpr> dim) : dim(std::move(dim)) {} \
        \
        virtual bool isSubtypeOf(const Type &type) const override { \
            const name ## Type &other = static_cast<const name ## Type &>(type); \
            return *dim == *other.dim; \
        } \
        virtual std::unique_ptr<Type> copy() const override { \
            std::unique_ptr<name ## Type> ret = std::make_unique<name ## Type>(std::move(dim->copy())); \
            extra_copy \
            return ret; \
        } \
        virtual bool isConstant() const override { \
            return dim->isConstant(); \
        } \
        virtual bool isCanonical() const override { \
            return !dim->isConstant() || dim->offset; \
        } \
        virtual std::string toString() const override { \
            if (dim->isConstant() && dim->offset == 1) { \
                return std::string(#name); \
            } else { \
                return std::string(#name "[") + dim->toString() + "]"; \
            } \
        } \
        virtual bool isClassical() const override { \
            return is_classical; \
        } \
        virtual bool isReversibleFriendly() const override { \
            return is_rev_friendly; \
        } \
        virtual bool isMaterializable() const override { \
            return is_materializable; \
        } \
        virtual std::unique_ptr<Type> canonicalize() const override { \
            if (isCanonical()) { \
                return std::move(copy()); \
            } else { \
                return std::make_unique<TupleType>(); \
            } \
        } \
        virtual std::unique_ptr<BroadcastType> collapseToBroadcast(const BroadcastType &genericType) const override { \
            return std::make_unique<BroadcastType>( \
                std::make_unique<name ## Type>(std::make_unique<DimVarExpr>("", 1)), \
                std::move(dim->copy())); \
        } \
        virtual void doInferDimvars(const Type &constType, DimVarValues &out) const override { \
            if (const name##Type *const##name = dynamic_cast<const name##Type *>(&constType)) { \
                return dim->inferDimvars(*const##name->dim, out); \
            } else { \
                throw DimVarInferenceTypeMismatch(constType, *this); \
            } \
        } \
        virtual void evalDimVarExprs(DimVarValues &dimvar_values, bool permissive) override { \
            dim->eval(dimvar_values, permissive); \
        } \
        virtual llvm::SmallVector<mlir::Type> toMlirType(MlirHandle &handle) const override { \
            assert(dim->isConstant() && "Can't convert non-constant type to MLIR"); \
            uint64_t dim_ = dim->offset; \
            (void)dim_; \
            return type_gen; \
        } \
    }

// Corresponds to qubit[N]
PSEUDO_ARR_TYPE(Qubit, /*is_classical=*/false, /*is_rev_friendly=*/false,
                /*is_materializable=*/true, copy(),
                {handle.builder.getType<qwerty::QBundleType>(dim_)},
                /* no extra fields */, /* no extra copying */);
// Corresponds to bit[N]
PSEUDO_ARR_TYPE(Bit, /*is_classical=*/true, /*is_rev_friendly=*/true,
                /*is_materializable=*/true,
                std::make_unique<QubitType>(std::move(dim->copy())),
                {handle.builder.getType<qwerty::BitBundleType>(dim_)},
                /* no extra fields */, /* no extra copying */);
// Corresponds to basis[N], although programmers should not be able to write
// that in the current implementation
PSEUDO_ARR_TYPE(Basis, /*is_classical=*/true, /*is_rev_friendly=*/false,
                /*is_materializable=*/false, copy(), {},
                SpanList span;,
                ret->span.append(span););

#undef PSEUDO_ARR_TYPE

// Common code for both number types
#define NUMBER_TYPE(name, is_materializable, buildercall) \
    struct name##Type : Type { \
        virtual bool isSubtypeOf(const Type &type) const override { return true; } \
        virtual std::unique_ptr<Type> copy() const override { \
            return std::make_unique<name##Type>(); \
        } \
        virtual bool isConstant() const override { return true; } \
        virtual bool isCanonical() const override { return true; } \
        virtual bool isClassical() const override { return true; } \
        virtual bool isReversibleFriendly() const override { return true; } \
        virtual bool isMaterializable() const override { return is_materializable; } \
        virtual std::unique_ptr<Type> canonicalize() const override { \
            return std::move(copy()); \
        } \
        virtual void doInferDimvars(const Type &constType, DimVarValues &out) const override {} \
        virtual std::string toString() const override { return #name; } \
        virtual void evalDimVarExprs(DimVarValues &dimvar_values, bool permissive) override {} \
        virtual llvm::SmallVector<mlir::Type> toMlirType(MlirHandle &handle) const override { \
            return {buildercall}; \
        } \
    };

// Currently unused and likely poorly supported
NUMBER_TYPE(Int, /*is_materializable=*/true, handle.builder.getI64Type())
// Called "angle" in the frontend (so you can say e.g. angle[3])
NUMBER_TYPE(Float, /*is_materializable=*/true, handle.builder.getF64Type())
// Called "ampl" for "amplitude" in the frontend (so you can say e.g. ampl[32])
NUMBER_TYPE(Complex, /*is_materializable=*/false,
            mlir::ComplexType::get(handle.builder.getF64Type()))

#undef NUMBER_TYPE

// A function type as typically defined in the lambda calculus, T1 → T2. See
// the comment for TupleType above for details on how this maps to programmers'
// type annotations.
struct FuncType : Type {
    std::unique_ptr<Type> lhs;
    std::unique_ptr<Type> rhs;
    bool is_rev;

    FuncType(std::unique_ptr<Type> lhs, std::unique_ptr<Type> rhs, bool is_rev)
            : lhs(std::move(lhs)), rhs(std::move(rhs)), is_rev(is_rev) {}

    virtual bool isSubtypeOf(const Type &type) const override {
        const FuncType &funcType = static_cast<const FuncType &>(type);
        // The subtype relation is contravariant for functions. See the S-Arrow
        // rule in Chapter 15 of TAPL.
        return *funcType.lhs <= *lhs && *rhs <= *funcType.rhs
               // tau_1 -->^L tau_2 <: tau_1 -->^R tau_2
               // iff R => L (R implies L)
               // (where "-->^x" denotes "-->" if x=0, and "--rev-->" if x=1,
               //  and <: is the subtype relation, see Chapter 15 of TAPL)
               && (!funcType.is_rev || is_rev);
    }
    virtual std::unique_ptr<Type> copy() const override {
        return std::make_unique<FuncType>(std::move(lhs->copy()),
                                          std::move(rhs->copy()),
                                          is_rev);
    }
    virtual bool isConstant() const override {
        return lhs->isConstant() && rhs->isConstant();
    }
    virtual bool isCanonical() const override {
        return lhs->isCanonical() && rhs->isCanonical();
    }
    virtual bool isFurled() const override {
        return lhs->isFurled() || rhs->isFurled();
    }
    virtual bool isClassical() const override { return true; }
    virtual bool isReversibleFriendly() const override { return false; }
    virtual bool isMaterializable() const override {
        return lhs->isMaterializable() && rhs->isMaterializable();
    }
    virtual std::unique_ptr<Type> asEmbedded(DebugInfo &dbg, EmbeddingKind kind) const override;
    virtual std::unique_ptr<Type> canonicalize() const override {
        return std::move(std::make_unique<FuncType>(std::move(lhs->canonicalize()),
                                                    std::move(rhs->canonicalize()),
                                                    is_rev));
    }
    virtual std::unique_ptr<Type> unfurl() const override {
        return std::move(std::make_unique<FuncType>(std::move(lhs->unfurl()),
                                                    std::move(rhs->unfurl()),
                                                    is_rev));
    }
    virtual void doInferDimvars(const Type &constType, DimVarValues &out) const override {
        const FuncType *constFuncType = dynamic_cast<const FuncType *>(&constType);
        if (constFuncType) {
            lhs->doInferDimvars(*constFuncType->lhs, out);
            rhs->doInferDimvars(*constFuncType->rhs, out);
        } else {
            throw DimVarInferenceTypeMismatch(constType, *this);
        }
    }
    virtual std::string toString() const override;
    virtual void evalDimVarExprs(DimVarValues &dimvar_values, bool permissive) override {
        lhs->evalDimVarExprs(dimvar_values, permissive);
        rhs->evalDimVarExprs(dimvar_values, permissive);
    }
    virtual llvm::SmallVector<mlir::Type> toMlirType(MlirHandle &handle) const override {
        llvm::SmallVector<mlir::Type> arg_types = lhs->toMlirType(handle);
        llvm::SmallVector<mlir::Type> ret_types = rhs->toMlirType(handle);
        return {handle.builder.getType<qwerty::FunctionType>(
            handle.builder.getFunctionType(arg_types, ret_types), is_rev)};
    }
    virtual std::vector<const Type *> children() const override {
        return {lhs.get(), rhs.get()};
    };
};

// Holds some captured bits
struct Bits : HybridObj {
    std::vector<bool> bits;
    BitType type;

    Bits(std::vector<bool> bits) : bits(bits), type(std::make_unique<DimVarExpr>("", bits.size())) {}

    virtual const Type &getType() const override { return type; }
    virtual std::unique_ptr<HybridObj> copyHybrid() const override {
        return std::make_unique<Bits>(bits);
    }
    virtual HybridObj::Hash getHash() const override;
    size_t getNumBits() { return bits.size(); }
    bool getBit(size_t idx) { return bits[idx]; }
};

// Holds a captured integer
struct Integer : HybridObj {
    DimVarValue val;
    IntType type;

    Integer(DimVarValue val) : val(val) {}

    virtual const Type &getType() const override { return type; }
    virtual std::unique_ptr<HybridObj> copyHybrid() const override {
        return std::make_unique<Integer>(val);
    }
    virtual HybridObj::Hash getHash() const override { return "Integer:" + std::to_string(val); }
};

// Holds a captured float
struct Angle : HybridObj {
    double val;
    FloatType type;

    Angle(double val) : val(val) {}

    virtual const Type &getType() const override { return type; }
    virtual std::unique_ptr<HybridObj> copyHybrid() const override {
        return std::make_unique<Angle>(val);
    }
    virtual HybridObj::Hash getHash() const override { return "Angle:" + std::to_string(val); }
};

// Holds a single complex number
struct Amplitude : HybridObj {
    std::complex<double> val;
    ComplexType type;

    Amplitude(std::complex<double> val) : val(val) {}

    virtual const Type &getType() const override { return type; }
    virtual std::unique_ptr<HybridObj> copyHybrid() const override {
        return std::make_unique<Amplitude>(val);
    }
    virtual HybridObj::Hash getHash() const override {
        return "Amplitude:" + std::to_string(val.real())
               + " + i*" + std::to_string(val.imag());
    }
};

// Holds a captured tuple that contains other HybridObjs. A captured angle[5]
// (whose Python object is really e.g. [0.4, 0.2, 0.0, 0.6, 0.9]) will be held
// using this object.
struct Tuple : HybridObj {
    std::unique_ptr<TupleType> type;
    std::vector<std::unique_ptr<HybridObj>> children;

    static std::unique_ptr<TupleType> calcType(
            std::vector<std::unique_ptr<HybridObj>> &kids) {
        std::vector<std::unique_ptr<Type>> child_types;
        for (const std::unique_ptr<HybridObj> &child : kids) {
            child_types.push_back(std::move(child->getType().copy()));
        }
        return std::make_unique<TupleType>(std::move(child_types));
    }

    Tuple(std::vector<std::unique_ptr<HybridObj>> children)
         : type(std::move(calcType(children))),
           children(std::move(children)) {}

    virtual const Type &getType() const override { return *type; }

    virtual std::unique_ptr<HybridObj> copyHybrid() const override {
        std::vector<std::unique_ptr<HybridObj>> copied_children;
        for (const std::unique_ptr<HybridObj> &child : children) {
            copied_children.push_back(std::move(child->copyHybrid()));
        }
        return std::make_unique<Tuple>(std::move(copied_children));
    }

    virtual HybridObj::Hash getHash() const override {
        std::ostringstream ss;
        ss << "(";
        for (const std::unique_ptr<HybridObj> &child : children) {
            ss << child->getHash();
            ss << ",";
        }
        ss << ")";
        return ss.str();
    }
};

// The abstract base class for a Qwerty AST node. This holds a lot of metadata
// for different stages of the compilation process. Perhaps the most important
// is the type, which subclasses are responsible for storing. The `metadata'
// field below also holds a union of many types of metadata, including MLIR
// values or netlist wires to which this node was lowered.
struct ASTNode {
    using Wires = std::vector<mockturtle::xag_network::signal>;
    using Values = llvm::SmallVector<mlir::Value>;
    struct BasisValue {
        qwerty::BasisAttr basis;
        Values phases;
    };

    // Each of these are used by different (or disjoint) stages of the
    // compiler. Might as well put them in a union{} to save a little memory
    std::variant<
        std::monostate,
        // For evaluating dimvarexprs
        DimVarValues,
        // For conversion to mockturtle xag
        Wires,
        // For lowering most Values to MLIR (qubits, functions, etc)
        Values,
        // For lowering bases to MLIR
        BasisValue
    > metadata;

    std::unique_ptr<DebugInfo> dbg;

    ASTNode(std::unique_ptr<DebugInfo> dbg)
           : dbg(std::move(dbg)) {}
    virtual ~ASTNode() {}

    virtual std::unique_ptr<ASTNode> copy() const = 0;

    // It's the subclass's job to store the type. However, before type
    // checking, there will not be a type. (And null references are not
    // possible in C++.) Hence the hasType() check.
    virtual bool hasType() const = 0;
    virtual const Type &getType() const = 0;
    // The node passed will always be exactly the same type as the subclass, so
    // it can be safely static_cast<>ed.
    virtual bool isEqual(const ASTNode &node) const = 0;

    bool operator==(const ASTNode &node) const {
        return typeid(*this) == typeid(node)
               && hasType() == node.hasType()
               && (!hasType() || getType() == node.getType())
               && *dbg == *node.dbg
               && isEqual(node);
    }
    bool operator!=(const ASTNode &node) const {
        return !(*this == node);
    }

    // Name of this AST node, printed out for debugging
    virtual std::string label() = 0;
    // Return a list of pairs of (edge_name, child_node). The edge_name is used
    // to annotate the edge drawn between parent and child when dumping out the
    // AST.
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() = 0;
    // Additional metadata to dump out when printing out the AST
    virtual std::vector<std::string> getAdditionalMetadata() const { return {}; };
    // Return a string representation of this AST node (useful for debugging).
    // The print_dbg determines whether full debug information is printed
    // (warning: this is very verbose)
    std::string dump(bool print_dbg);
    std::string dump() { return dump(false); }
    // Used for double dispatch, see the comment for the VISITOR_TEDIUM macro
    // below.
    virtual bool visit(ASTVisitContext &ctx, ASTVisitor &visitor) = 0;

    // "getX()" will create the metadata if it doesn't already exist (and trash
    // any existing metadata). "peekX()" does not modify any metadata and
    // returns NULL instead if the metadata does not exist
    #define METADATA_ACCESSOR(type, name) \
        inline type &get##name() { \
            if (std::holds_alternative<type>(metadata)) { \
                return std::get<type>(metadata); \
            } else { \
                return metadata.emplace<type>(); \
            } \
        } \
        inline const type *peek##name() const { \
            if (std::holds_alternative<type>(metadata)) { \
                return &std::get<type>(metadata); \
            } else { \
                return nullptr; \
            } \
        }

    METADATA_ACCESSOR(DimVarValues, ScopedDimvars);
    METADATA_ACCESSOR(Wires, Wires);
    METADATA_ACCESSOR(Values, MlirValues);
    METADATA_ACCESSOR(BasisValue, Basis);

    #undef METADATA_ACCESSOR

    // Currently this scopedDimvar stuff is use for loop variables, like the
    // `i' in `for i in range(N)'.
    void setScopedDimvar(DimVar dimvar, DimVarValue val) {
        getScopedDimvars()[dimvar] = val;
    }
    void inheritScopedDimvars(const ASTNode &other) {
        // Delicate: when we inherit, we should prefer our own
        if (const DimVarValues *other_dvs = other.peekScopedDimvars()) {
            DimVarValues new_dvs = *other_dvs;
            DimVarValues &my_dvs = getScopedDimvars();
            new_dvs.insert(my_dvs.begin(), my_dvs.end());
            getScopedDimvars() = new_dvs;
        } else {
            // Nothing to do. Parent does not have scoped dimvars so we have
            // nothing to change
        }
    }
    void copyInternalsFrom(const ASTNode &other) {
        metadata = other.metadata;
    }
    void walk(ASTVisitContext &ctx, ASTVisitor &visitor);
    // For root node, use myself as parent
    void walk(ASTVisitor &visitor);
};

// God bless this country
// https://en.wikipedia.org/wiki/Double_dispatch#Double_dispatch_in_C++
#define VISITOR_TEDIUM \
    virtual bool visit(ASTVisitContext &ctx, ASTVisitor &visitor) override { \
        return visitor.visit(ctx, *this); \
    }

struct Expr : ASTNode {
    Expr(std::unique_ptr<DebugInfo> dbg)
        : ASTNode(std::move(dbg)) {}
};

// Believe it or not, this is a variable.
// Example syntax:
//     my_variable
struct Variable : Expr {
    std::string name;
    std::unique_ptr<Type> type;

    Variable(std::unique_ptr<DebugInfo> dbg,
             const std::string name)
            : Expr(std::move(dbg)),
              name(name) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Variable>(std::move(dbg->copy()),
                                           name);
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Variable &var = static_cast<const Variable &>(node);
        return name == var.name;
    }
    virtual std::string label() override { return "Variable"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    VISITOR_TEDIUM
};

// A classical embedding.
// Example syntax:
//     f.sign
struct EmbedClassical : Expr {
    std::string name;
    std::string operand_name;
    EmbeddingKind kind;
    std::unique_ptr<Type> type;

    EmbedClassical(std::unique_ptr<DebugInfo> dbg,
                   const std::string name,
                   const std::string operand_name,
                   EmbeddingKind kind)
                  : Expr(std::move(dbg)),
                    name(name),
                    operand_name(operand_name),
                    kind(kind) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result = std::make_unique<EmbedClassical>(
                std::move(dbg->copy()), name, operand_name, kind);
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const EmbedClassical &embed = static_cast<const EmbedClassical &>(node);
        return name == embed.name
               && operand_name == embed.operand_name
               && kind == embed.kind;
    }
    virtual std::string label() override { return "EmbedClassical"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    VISITOR_TEDIUM
};

// A function that initializes qubits in the state specified by a (singleton)
// basis operand or a bit[N].
//
// Example syntax:
//     {'+'}.prep
// The guarantee is that '0' | {'+'}.prep will evaluate to '+'.
//
// Another example:
//     bit[4](0b1101).prep
// The guarantee is that '0000' | bit[4](0b1101).prep will evaluate to '1101'.
struct Prepare : Expr {
    std::unique_ptr<ASTNode> operand;
    std::unique_ptr<Type> type;

    Prepare(std::unique_ptr<DebugInfo> dbg,
            std::unique_ptr<ASTNode> operand)
           : Expr(std::move(dbg)),
             operand(std::move(operand)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result = std::make_unique<Prepare>(
                std::move(dbg->copy()), std::move(operand->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Prepare &prep = static_cast<const Prepare &>(node);
        return *operand == *prep.operand;
    }
    virtual std::string label() override { return "Prepare"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"operand", operand}};
    }
    VISITOR_TEDIUM
};

// Lift some bit[N] or ampl[M] to a qubit state.
// Example syntax:
//     bit[4](0b1101).q
// which evaluates to '1101'. This example is silly but it's useful to prepare
// a captured bit[N] as qubits.
//
// Another example, if `capture' is a Python `list' of `float's [1.4142, 1.4142]:
//     capture.q
// which evaluates to 'p'.
struct Lift : Expr {
    std::unique_ptr<ASTNode> operand;
    std::unique_ptr<Type> type;

    Lift(std::unique_ptr<DebugInfo> dbg,
             std::unique_ptr<ASTNode> operand)
            : Expr(std::move(dbg)),
              operand(std::move(operand)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result = std::make_unique<Lift>(
                std::move(dbg->copy()), std::move(operand->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Lift &lift = static_cast<const Lift &>(node);
        return *operand == *lift.operand;
    }
    virtual std::string label() override { return "Lift"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"operand", operand}};
    }
    VISITOR_TEDIUM
};

// Take the adjoint of a function value.
// Example syntax:
//     ~f
struct Adjoint : Expr {
    std::unique_ptr<ASTNode> operand;
    std::unique_ptr<Type> type;

    Adjoint(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> operand)
          : Expr(std::move(dbg)),
            operand(std::move(operand)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result = std::make_unique<Adjoint>(
                std::move(dbg->copy()), std::move(operand->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Adjoint &adj = static_cast<const Adjoint &>(node);
        return *operand == *adj.operand;
    }
    virtual std::string label() override { return "Adjoint"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"operand", operand}};
    }
    VISITOR_TEDIUM
};

// Call a function value.
// Example syntax:
//     '0' | flip
//         ^
//          \ this guy
struct Pipe : Expr {
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    std::unique_ptr<Type> type;

    Pipe(std::unique_ptr<DebugInfo> dbg,
         std::unique_ptr<ASTNode> left,
         std::unique_ptr<ASTNode> right)
        : Expr(std::move(dbg)),
          left(std::move(left)),
          right(std::move(right)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Pipe>(std::move(dbg->copy()),
                                       std::move(left->copy()),
                                       std::move(right->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Pipe &pipe = static_cast<const Pipe &>(node);
        return *left == *pipe.left
               && *right == *pipe.right;
    }
    virtual std::string label() override { return "Pipe"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"left", left}, {"right", right}};
    }
    VISITOR_TEDIUM
};

// Instantiate a capture that had a free dimension variable by specifing a
// value for that variable.
// Example syntax:
//     my_capture[[42]]
struct Instantiate : Expr {
    std::unique_ptr<ASTNode> var;
    std::vector<std::unique_ptr<DimVarExpr>> instance_vals;
    std::unique_ptr<Type> type;

    Instantiate(std::unique_ptr<DebugInfo> dbg,
         std::unique_ptr<ASTNode> var,
         std::vector<std::unique_ptr<DimVarExpr>> instance_vals)
        : Expr(std::move(dbg)),
          var(std::move(var)),
          instance_vals(std::move(instance_vals)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Instantiate>(std::move(dbg->copy()),
                                       std::move(var->copy()),
                                       std::move(copy_vector_of_copyable(instance_vals)));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Instantiate &instantiate = static_cast<const Instantiate &>(node);
        return *var == *instantiate.var
               && compare_vectors(instance_vals, instantiate.instance_vals);
    }
    virtual std::string label() override { return "Instantiate"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"var", var}};
    }
    VISITOR_TEDIUM
};

// Apply a function multiple times
// Example syntax:
//     '0' | (flip for i in range(3)) | measure
// This flips the bit 3 times.
struct Repeat : Expr {
    std::unique_ptr<ASTNode> body;
    std::string loopvar;
    std::unique_ptr<DimVarExpr> ub;
    std::unique_ptr<Type> type;

    Repeat(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> body,
           std::string loopvar,
           std::unique_ptr<DimVarExpr> ub)
          : Expr(std::move(dbg)),
            body(std::move(body)),
            loopvar(loopvar),
            ub(std::move(ub)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Repeat>(std::move(dbg->copy()),
                                         std::move(body->copy()),
                                         loopvar,
                                         std::move(ub->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Repeat &rep = static_cast<const Repeat &>(node);
        return *body == *rep.body
               && loopvar == loopvar
               && *ub == *rep.ub;
    }
    virtual std::string label() override { return "Repeat"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"body", body}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Loop variable: " + loopvar,
                "Upper bound: " + ub->toString()};
    };
    VISITOR_TEDIUM
};

// Tensor an expression with itself multiple times
// Example syntax:
//     ['0' for i in range(5)]
// This example is equivalent to '0'+'0'+'0'+'0'+'0'.
struct RepeatTensor : Expr {
    std::unique_ptr<ASTNode> body;
    std::string loopvar;
    std::unique_ptr<DimVarExpr> ub;
    std::unique_ptr<Type> type;

    RepeatTensor(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> body,
           std::string loopvar,
           std::unique_ptr<DimVarExpr> ub)
          : Expr(std::move(dbg)),
            body(std::move(body)),
            loopvar(loopvar),
            ub(std::move(ub)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
            std::make_unique<RepeatTensor>(std::move(dbg->copy()),
                                           std::move(body->copy()),
                                           loopvar,
                                           std::move(ub->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const RepeatTensor &rep = static_cast<const RepeatTensor &>(node);
        return *body == *rep.body
               && loopvar == loopvar
               && *ub == *rep.ub;
    }
    virtual std::string label() override { return "RepeatTensor"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"body", body}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Loop variable: " + loopvar,
                "Upper bound: " + ub->toString()};
    };
    VISITOR_TEDIUM
};

// For programmer convenience, you can write b & f _or_ you can write f & b.
// (Here, f is a reversible function and b is a basis.) However, at the time
// convert_ast.py creates this AST node, it does not know which operand is
// which. Only during type checking do we figure this out. So the order begins
// with UNKNOWN and then type checking sets it to B_U or U_B (here, "U" means
// the Unitary represented by the reversible function).
typedef enum PredOrder {
    PRED_ORDER_B_U,
    PRED_ORDER_U_B,
    PRED_ORDER_UNKNOWN
} PredOrder;

// Predicate a function, i.e., force it to run only in a (proper) subspace.
// Example syntax:
//     {'1'} & flip
// This particular example is equivalent to the well-known CNOT gate.
struct Pred : Expr {
    PredOrder order;
    std::unique_ptr<ASTNode> basis;
    std::unique_ptr<ASTNode> body;
    std::unique_ptr<Type> type;

    Pred(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> basis,
           std::unique_ptr<ASTNode> body)
          : Expr(std::move(dbg)),
            order(PRED_ORDER_UNKNOWN),
            basis(std::move(basis)),
            body(std::move(body)) {}

    Pred(std::unique_ptr<DebugInfo> dbg,
           PredOrder order,
           std::unique_ptr<ASTNode> basis,
           std::unique_ptr<ASTNode> body)
          : Expr(std::move(dbg)),
            order(order),
            basis(std::move(basis)),
            body(std::move(body)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<Pred> result =
                std::make_unique<Pred>(std::move(dbg->copy()),
                                       std::move(basis->copy()),
                                       std::move(body->copy()));
        result->order = order;
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Pred &pred = static_cast<const Pred &>(node);
        return order == pred.order
               && *basis == *pred.basis
               && *body == *pred.body;
    }
    virtual std::string label() override { return "Pred"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        switch (order) {
        case PRED_ORDER_UNKNOWN:
            return {{"left", basis}, {"right", body}};
        case PRED_ORDER_B_U:
            return {{"basis", basis}, {"body", body}};
        case PRED_ORDER_U_B:
            return {{"body", body}, {"basis", basis}};
        default:
            assert(0 && "Missing handling for PredOrder");
            return {};
        }
    }
    VISITOR_TEDIUM
};

// The tensor product of two terms.
// Example syntax:
//     '0' + '1'
struct BiTensor : Expr {
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    std::unique_ptr<Type> type;
    // Metadata used for typechecking basis literals and .prep, respectively
    bool only_literals, singleton_basis;

    BiTensor(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> left,
           std::unique_ptr<ASTNode> right)
          : Expr(std::move(dbg)),
            left(std::move(left)),
            right(std::move(right)),
            only_literals(false),
            singleton_basis(false) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BiTensor>(std::move(dbg->copy()),
                                           std::move(left->copy()),
                                           std::move(right->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BiTensor &bi = static_cast<const BiTensor &>(node);
        return *left == *bi.left
               && *right == *bi.right;
    }
    virtual std::string label() override { return "BiTensor"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"left", left}, {"right", right}};
    }
    VISITOR_TEDIUM
};

// An n-fold tensor product of a term.
// Example syntax:
//     '10'[3]
// This example is equivalent to '10'+'10'+'10'.
struct BroadcastTensor : Expr {
    // '10'[3]
    // \__/ ^
    //  |    \____ factor
    // value
    std::unique_ptr<ASTNode> value;
    std::unique_ptr<DimVarExpr> factor;
    std::unique_ptr<Type> type;
    // Metadata used for typechecking basis literals and .prep, respectively
    bool only_literals, singleton_basis;

    BroadcastTensor(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> value,
           std::unique_ptr<DimVarExpr> factor)
          : Expr(std::move(dbg)),
            value(std::move(value)),
            factor(std::move(factor)),
            only_literals(false),
            singleton_basis(false) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BroadcastTensor>(std::move(dbg->copy()),
                                                  std::move(value->copy()),
                                                  std::move(factor->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BroadcastTensor &broad = static_cast<const BroadcastTensor &>(node);
        return *value == *broad.value
               && *factor == *broad.factor;
    }
    virtual std::string label() override { return "BroadcastTensor"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"value", value}};
    }
    VISITOR_TEDIUM
};

// A qubit literal.
// Example syntax:
//     '0'[5]
struct QubitLiteral : Expr {
    // For example, '1'[N] would be:
    //   eigenstate = MINUS
    //   prim_basis = Z
    //   dim = N
    Eigenstate eigenstate;
    PrimitiveBasis prim_basis;
    QubitType type;

    QubitLiteral(std::unique_ptr<DebugInfo> dbg,
                 Eigenstate eigenstate,
                 PrimitiveBasis prim_basis,
                 std::unique_ptr<DimVarExpr> dim)
                : Expr(std::move(dbg)),
                  eigenstate(eigenstate),
                  prim_basis(prim_basis),
                  type(std::move(dim)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<QubitLiteral>(std::move(dbg->copy()),
                                               eigenstate,
                                               prim_basis,
                                               std::move(type.dim->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override {
        return type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const QubitLiteral &qlit = static_cast<const QubitLiteral &>(node);
        return eigenstate == qlit.eigenstate
               && prim_basis == qlit.prim_basis;
    }
    virtual std::string label() override { return "QubitLiteral"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override;
    VISITOR_TEDIUM
};

// Impart a phase shift, i.e., a scalar factor of e^{i\theta} on either a
// reversible function or a qubit.
// Example syntax:
//     -'1'
//     ^
//      \ this guy
//
// Another example:
//     '1' @ 180
//         \____/
//            \ this guy
struct Phase : Expr {
    // The phase angle theta
    std::unique_ptr<ASTNode> phase;
    // The thing being rotated ('1' in the example above)
    std::unique_ptr<ASTNode> value;
    std::unique_ptr<Type> type;
    // Metadata used for typechecking basis literal
    bool only_literals;

    Phase(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> phase,
           std::unique_ptr<ASTNode> value)
         : Expr(std::move(dbg)),
           phase(std::move(phase)),
           value(std::move(value)),
           only_literals(false) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Phase>(std::move(dbg->copy()),
                                        std::move(phase->copy()),
                                        std::move(value->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Phase &ph = static_cast<const Phase &>(node);
        return *phase == *ph.phase
               && *value == *ph.value;
    }
    virtual std::string label() override { return "Phase"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"phase", phase},
                {"value", value}};
    }
    VISITOR_TEDIUM
};

// A superposition of qubit literals.
// Example syntax:
//     0.25*'0p' or 0.75*-'1m'
// This prepares the state (1/2)|0⟩|+⟩ + (√3/2)|1⟩|-⟩
struct SuperposLiteral : Expr {
    std::vector<std::pair<double, std::unique_ptr<ASTNode>>> pairs;
    std::unique_ptr<Type> type;

    SuperposLiteral(std::unique_ptr<DebugInfo> dbg,
                 std::vector<std::pair<double, std::unique_ptr<ASTNode>>> pairs)
                : Expr(std::move(dbg)),
                  pairs(std::move(pairs)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<SuperposLiteral>(
                    std::move(dbg->copy()),
                    std::move(copy_vector_of_copyable_pair<double, ASTNode>(pairs)));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const SuperposLiteral &split = static_cast<const SuperposLiteral &>(node);
        return compare_vectors_of_pairs(pairs, split.pairs);
    }
    virtual std::string label() override { return "SuperposLiteral"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override;
    VISITOR_TEDIUM
};

// A float constant.
// Example syntax:
//     3.14
struct FloatLiteral : Expr {
    double value;
    FloatType type;

    FloatLiteral(std::unique_ptr<DebugInfo> dbg,
                 double value)
                : Expr(std::move(dbg)),
                  value(value) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<FloatLiteral>(std::move(dbg->copy()),
                                               value);
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const FloatLiteral &flit = static_cast<const FloatLiteral &>(node);
        return value == flit.value;
    }
    virtual std::string label() override { return "FloatLiteral"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Value: " + std::to_string(value)};
    };
    VISITOR_TEDIUM
};

// A unary negation of a float.
// Example syntax:
//    -3.14
//    ^
//     \__ this guy
struct FloatNeg : Expr {
    std::unique_ptr<ASTNode> operand;
    FloatType type;

    FloatNeg(std::unique_ptr<DebugInfo> dbg,
             std::unique_ptr<ASTNode> operand)
            : Expr(std::move(dbg)),
              operand(std::move(operand)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<FloatNeg>(std::move(dbg->copy()),
                                           std::move(operand->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const FloatNeg &fneg = static_cast<const FloatNeg &>(node);
        return *operand == *fneg.operand;
    }
    virtual std::string label() override { return "FloatNeg"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"operand", operand}};
    }
    VISITOR_TEDIUM
};

// A binary operation between floats, usually for calculating rotation/phase
// angles.
// Example syntax:
//     3.14 + 0.0
//          ^
//           \____ this guy
struct FloatBinaryOp : Expr {
    FloatOp op;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    FloatType type;

    FloatBinaryOp(std::unique_ptr<DebugInfo> dbg,
                  FloatOp op,
                  std::unique_ptr<ASTNode> left,
                  std::unique_ptr<ASTNode> right)
                 : Expr(std::move(dbg)),
                   op(op),
                   left(std::move(left)),
                   right(std::move(right)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<FloatBinaryOp>(std::move(dbg->copy()),
                                                op,
                                                std::move(left->copy()),
                                                std::move(right->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const FloatBinaryOp &fbin = static_cast<const FloatBinaryOp &>(node);
        return op == fbin.op
               && *left == *fbin.left
               && *right == *fbin.right;
    }
    virtual std::string label() override { return "FloatBinaryOp"; }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Operation: " + float_op_name(op)};
    };
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"left", left}, {"right", right}};
    }
    VISITOR_TEDIUM
};

// A dimension variable expression used inside of a float expression.
// Example syntax:
//     N/2.0
// The evaluation of constant expressions makes this a constant, so then it can
// be canonicalized to a FloatLiteral node.
struct FloatDimVarExpr : Expr {
    std::unique_ptr<DimVarExpr> value;
    FloatType type;

    FloatDimVarExpr(std::unique_ptr<DebugInfo> dbg,
                   std::unique_ptr<DimVarExpr> value)
                  : Expr(std::move(dbg)),
                    value(std::move(value)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<FloatDimVarExpr>(std::move(dbg->copy()),
                                                 std::move(value->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const FloatDimVarExpr &fdve = static_cast<const FloatDimVarExpr &>(node);
        return *value == *fdve.value;
    }
    virtual std::string label() override { return "FloatDimVarExpr"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    VISITOR_TEDIUM
};

// A tuple literal.
// Example syntax:
//     (2.3, 5.4)
struct TupleLiteral : Expr {
    std::vector<std::unique_ptr<ASTNode>> elts;
    std::unique_ptr<Type> type;

    // Unit literal
    TupleLiteral(std::unique_ptr<DebugInfo> dbg) : Expr(std::move(dbg)) {}
    TupleLiteral(std::unique_ptr<DebugInfo> dbg,
                 std::vector<std::unique_ptr<ASTNode>> elts)
                : Expr(std::move(dbg)),
                  elts(std::move(elts)) {}
    bool isUnit() const { return elts.empty(); }
    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<TupleLiteral>(std::move(dbg->copy()),
                                               std::move(copy_vector_of_copyable<ASTNode>(elts)));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const TupleLiteral &tup = static_cast<const TupleLiteral &>(node);
        return compare_vectors(elts, tup.elts);
    }
    virtual std::string label() override { return "TupleLiteral"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override;
    VISITOR_TEDIUM
};

// A primitive basis with a dimension. See Section 2.2 of the CGO paper.
// Example syntax:
//     pm[5]
struct BuiltinBasis : Expr {
    // For example, std[N] would be:
    //   prim_basis = Z
    //   dim (of the BasisType) = N
    PrimitiveBasis prim_basis;
    BasisType type;

    BuiltinBasis(std::unique_ptr<DebugInfo> dbg,
                  PrimitiveBasis prim_basis,
                  std::unique_ptr<DimVarExpr> dim)
                 : Expr(std::move(dbg)),
                   prim_basis(prim_basis),
                   type(std::move(dim)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BuiltinBasis>(std::move(dbg->copy()),
                                                prim_basis,
                                                std::move(type.dim->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const BuiltinBasis &std = static_cast<const BuiltinBasis &>(node);
        return prim_basis == std.prim_basis;
    }
    virtual std::string label() override { return "BuiltinBasis"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override;
    VISITOR_TEDIUM
};

// The identity operation on qubits.
// Example syntax:
//     id[3]
struct Identity : Expr {
    FuncType type;

    static FuncType createType(std::unique_ptr<DimVarExpr> dim) {
        std::unique_ptr<DimVarExpr> dim_copy = dim->copy();
        return std::move(FuncType(std::make_unique<QubitType>(std::move(dim_copy)),
                                  std::make_unique<QubitType>(std::move(dim)),
                                  /*is_rev=*/true));
    }

    Identity(std::unique_ptr<DebugInfo> dbg,
             std::unique_ptr<DimVarExpr> dim)
            : Expr(std::move(dbg)),
              type(createType(std::move(dim))) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Identity>(std::move(dbg->copy()),
                                           std::move(static_cast<QubitType &>(*type.lhs).dim->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override { return true; }
    virtual std::string label() override { return "Identity"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    VISITOR_TEDIUM
};

// The mighty basis translation. (See Section 2.2 of the CGO paper.)
// Example syntax:
//     {'0','1'} >> {'0',-'1'}
struct BasisTranslation : Expr {
    std::unique_ptr<ASTNode> basis_in;
    std::unique_ptr<ASTNode> basis_out;
    std::unique_ptr<Type> type;

    BasisTranslation(std::unique_ptr<DebugInfo> dbg,
                     std::unique_ptr<ASTNode> basis_in,
                     std::unique_ptr<ASTNode> basis_out)
                    : Expr(std::move(dbg)),
                      basis_in(std::move(basis_in)),
                      basis_out(std::move(basis_out)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BasisTranslation>(std::move(dbg->copy()),
                                                   std::move(basis_in->copy()),
                                                   std::move(basis_out->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const BasisTranslation &trans = static_cast<const BasisTranslation &>(node);
        return *basis_in == *trans.basis_in
               && *basis_out == *trans.basis_out;
    }
    virtual std::string label() override { return "BasisTranslation"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"basis_in", basis_in},
                {"basis_out", basis_out}};
    }
    VISITOR_TEDIUM
};

// Discard qubits, resetting them back to the |0⟩ state afterward.
// Example syntax:
//     discard[3]
struct Discard : Expr {
    FuncType type;

    Discard(std::unique_ptr<DebugInfo> dbg,
            std::unique_ptr<DimVarExpr> dim)
           : Expr(std::move(dbg)),
             type(std::move(std::make_unique<QubitType>(std::move(dim))),
                  std::move(std::make_unique<TupleType>()),
                  /*is_rev=*/false) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Discard>(std::move(dbg->copy()),
                                          std::move(static_cast<QubitType &>(*type.lhs).dim->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override { return type; }
    virtual bool isEqual(const ASTNode &node) const override { return true; }
    virtual std::string label() override { return "Discard"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override { return {}; }
    VISITOR_TEDIUM
};

// Measure qubits and return measurement results as bits.
// Example syntax:
//     std[5].measure
struct Measure : Expr {
    std::unique_ptr<ASTNode> basis;
    std::unique_ptr<Type> type;

    Measure(std::unique_ptr<DebugInfo> dbg,
            std::unique_ptr<ASTNode> basis)
           : Expr(std::move(dbg)),
             basis(std::move(basis)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Measure>(std::move(dbg->copy()),
                                          std::move(basis->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Measure &meas = static_cast<const Measure &>(node);
        return *basis == *meas.basis;
    }
    virtual std::string label() override { return "Measure"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"basis", basis}};
    }
    VISITOR_TEDIUM
};

// Measure a qubit but return the qubit instead of the measurement result
// Example syntax:
//     std[5].proj
struct Project : Expr {
    std::unique_ptr<ASTNode> basis;
    std::unique_ptr<Type> type;

    Project(std::unique_ptr<DebugInfo> dbg,
            std::unique_ptr<ASTNode> basis)
           : Expr(std::move(dbg)),
             basis(std::move(basis)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Project>(std::move(dbg->copy()),
                                          std::move(basis->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Project &proj = static_cast<const Project &>(node);
        return *basis == *proj.basis;
    }
    virtual std::string label() override { return "Project"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"basis", basis}};
    }
    VISITOR_TEDIUM
};

// Syntactic sugar for swapping two basis vectors.
// Example syntax:
//     pm.flip
// This example is equivalent to {'p','m'} >> {'m','p'}
struct Flip : Expr {
    std::unique_ptr<ASTNode> basis;
    std::unique_ptr<Type> type;

    Flip(std::unique_ptr<DebugInfo> dbg,
         std::unique_ptr<ASTNode> basis)
        : Expr(std::move(dbg)),
          basis(std::move(basis)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Flip>(std::move(dbg->copy()),
                                       std::move(basis->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Flip &flip = static_cast<const Flip &>(node);
        return *basis == *flip.basis;
    }
    virtual std::string label() override { return "Flip"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"basis", basis}};
    }
    VISITOR_TEDIUM
};

// Syntactic sugar for a Bloch sphere rotation.
// Example syntax:
//     pm.rotate(pi/4)
struct Rotate : Expr {
    std::unique_ptr<ASTNode> basis;
    std::unique_ptr<ASTNode> theta;
    std::unique_ptr<Type> type;

    Rotate(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> basis,
           std::unique_ptr<ASTNode> theta)
          : Expr(std::move(dbg)),
            basis(std::move(basis)),
            theta(std::move(theta)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Rotate>(std::move(dbg->copy()),
                                         std::move(basis->copy()),
                                         std::move(theta->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Rotate &rot = static_cast<const Rotate &>(node);
        return *basis == *rot.basis
               && *theta == *rot.theta;
    }
    virtual std::string label() override { return "Rotate"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"basis", basis}, {"theta", theta}};
    }
    VISITOR_TEDIUM
};

// A user-provided basis.
// Example syntax:
//     {'00', '10', '01', '11'}
struct BasisLiteral : Expr {
    std::vector<std::unique_ptr<ASTNode>> elts;
    std::unique_ptr<Type> type;

    BasisLiteral(std::unique_ptr<DebugInfo> dbg,
                 std::vector<std::unique_ptr<ASTNode>> elts)
                : Expr(std::move(dbg)),
                  elts(std::move(elts)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BasisLiteral>(std::move(dbg->copy()),
                                               std::move(copy_vector_of_copyable<ASTNode>(elts)));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const BasisLiteral &blit = static_cast<const BasisLiteral &>(node);
        return compare_vectors(elts, blit.elts);
    }
    virtual std::string label() override { return "BasisLiteral"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override;
    VISITOR_TEDIUM
};

// A conditional expression that performs classical branching.
// Example syntax:
//     '0' | (flip if measure('+') else id) | measure
// (This example is a very roundabout way of getting a random classical bit.)
struct Conditional : Expr {
    std::unique_ptr<ASTNode> if_expr;
    std::unique_ptr<ASTNode> then_expr;
    std::unique_ptr<ASTNode> else_expr;
    std::unique_ptr<Type> type;

    Conditional(std::unique_ptr<DebugInfo> dbg,
                std::unique_ptr<ASTNode> if_expr,
                std::unique_ptr<ASTNode> then_expr,
                std::unique_ptr<ASTNode> else_expr)
              : Expr(std::move(dbg)),
                if_expr(std::move(if_expr)),
                then_expr(std::move(then_expr)),
                else_expr(std::move(else_expr)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
            std::make_unique<Conditional>(std::move(dbg->copy()),
                                          std::move(if_expr->copy()),
                                          std::move(then_expr->copy()),
                                          std::move(else_expr->copy()));

        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Conditional &cond = static_cast<const Conditional &>(node);
        return *if_expr == *cond.if_expr
               && *then_expr == *cond.then_expr
               && *else_expr == *cond.else_expr;
    }
    virtual std::string label() override { return "Conditional"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"if_expr", if_expr},
                {"then_expr", then_expr},
                {"else_expr", else_expr}};
    }
    VISITOR_TEDIUM
};

// Slicing bit[N] or tuples in typical Python fashion.
// Example syntax:
//     skip_the_first_bit_of_this_guy[1:]
struct Slice : Expr {
    std::unique_ptr<ASTNode> val;
    std::unique_ptr<DimVarExpr> lower;
    std::unique_ptr<DimVarExpr> upper;
    std::unique_ptr<Type> type;

    Slice(std::unique_ptr<DebugInfo> dbg,
          std::unique_ptr<ASTNode> val,
          std::unique_ptr<DimVarExpr> lower,
          std::unique_ptr<DimVarExpr> upper)
         : Expr(std::move(dbg)),
           val(std::move(val)),
           lower(std::move(lower)),
           upper(std::move(upper)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Slice>(std::move(dbg->copy()),
                                        std::move(val->copy()),
                                        lower? std::move(lower->copy()) : nullptr,
                                        upper? std::move(upper->copy()) : nullptr);
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override { return *type; }
    virtual bool isEqual(const ASTNode &node) const override {
        const Slice &slice = static_cast<const Slice &>(node);
        return val == slice.val
               && lower == slice.lower
               && upper == slice.upper;
    }
    virtual std::string label() override { return "Slice"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"val", val}};
    }
    VISITOR_TEDIUM
};

struct Stmt : ASTNode {
    Stmt(std::unique_ptr<DebugInfo> dbg)
        : ASTNode(std::move(dbg)) {}
};

// An assignment statement.
// Example syntax:
//     q = '0'
struct Assign : Stmt {
    std::string target;
    std::unique_ptr<ASTNode> value;
    std::unique_ptr<Type> type;

    Assign(std::unique_ptr<DebugInfo> dbg,
           std::string id,
           std::unique_ptr<ASTNode> value)
          : Stmt(std::move(dbg)),
            target(id),
            value(std::move(value)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Assign>(std::move(dbg->copy()),
                                         target,
                                         std::move(value->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }

    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }

    virtual bool isEqual(const ASTNode &node) const override {
        const Assign &assign = static_cast<const Assign &>(node);
        return target == assign.target && *value == *assign.value;
    }

    virtual std::string label() override { return "Assign"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"value", value}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Target: " + target};
    }
    VISITOR_TEDIUM
};

// A destructuring assignment statement.
// Example syntax:
//     q1, q2 = '01'
// This example is equivalent to:
//     q1 = '0'
//     q2 = '1'
struct DestructAssign : Stmt {
    std::vector<std::string> targets;
    std::unique_ptr<ASTNode> value;
    std::unique_ptr<Type> type;

    DestructAssign(std::unique_ptr<DebugInfo> dbg,
                   std::vector<std::string> targets,
                   std::unique_ptr<ASTNode> value)
                  : Stmt(std::move(dbg)),
                    targets(targets),
                    value(std::move(value)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<DestructAssign>(
                    std::move(dbg->copy()),
                    targets,
                    std::move(value->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }

    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }

    virtual bool isEqual(const ASTNode &node) const override {
        const DestructAssign &assign = static_cast<const DestructAssign &>(node);
        return targets == assign.targets && *value == *assign.value;
    }

    virtual std::string label() override { return "DestructAssign"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"value", value}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        std::vector<std::string> lines{"Targets:"};
        lines.insert(lines.end(), targets.begin(), targets.end());
        return lines;
    }
    VISITOR_TEDIUM
};

// A return statement.
// Example syntax:
//     return measure('+')
struct Return : Stmt {
    std::unique_ptr<ASTNode> value;
    std::unique_ptr<Type> type;

    Return(std::unique_ptr<DebugInfo> dbg,
           std::unique_ptr<ASTNode> value)
          : Stmt(std::move(dbg)),
            value(std::move(value)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<Return>(std::move(dbg->copy()),
                                         std::move(value->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Return &ret = static_cast<const Return &>(node);
        return *value == *ret.value;
    }
    virtual std::string label() override { return "Return"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"value", value}};
    }
    VISITOR_TEDIUM
};

// A histogram of results across many shots: the key is a measurement result
// and the value is the number of times it was observed.
using KernelResult = std::unordered_map<std::vector<bool>, size_t>;
using CaptureInstantiates = std::vector<std::unique_ptr<std::map<const std::vector<DimVarValue>, std::unique_ptr<HybridObj>>>>;

// Parent class of both @qpu and @classical kernels.
// Example syntax:
//     @qpu
//     def randbit() -> bit:
//         return 'p' | measure
struct Kernel : ASTNode, HybridObj {
    std::string name;
    std::unique_ptr<Type> type;
    std::vector<std::string> capture_names;
    std::vector<std::unique_ptr<Type>> capture_types;
    std::vector<size_t> capture_freevars;
    std::vector<std::string> arg_names;
    std::vector<DimVar> dimvars;
    std::vector<std::unique_ptr<ASTNode>> body;
    // Filled in by Python frontend. Contains the unique func_id (aka pointer
    // to bytecode) and "generation" (recompile count)
    std::string unique_gen_id;
    // Filled in by registerExplicitDimvars()
    std::vector<std::string> explicit_dimvars;
    // Filled in by inferDimvars()
    std::vector<std::unique_ptr<HybridObj>> capture_objs;
    // Map is a unique_ptr because of MSVC shortcoming:
    // https://stackoverflow.com/q/60201465/321301
    CaptureInstantiates capture_instances;
    DimVarValues dimvar_values;
    bool needs_explicit_dimvars;
    std::vector<DimVar> missing_dimvars;
    // The fields below are filled in after constructor by compile()
    std::string funcOp_name;
    bool funcOp_private;
    qwerty::FuncOp funcOp;
    // This is a trick to compare captures without needing to hold a pointer to
    // the previous captures
    std::vector<HybridObj::Hash> capture_hashes;

    Kernel(std::unique_ptr<DebugInfo> dbg,
           std::string name,
           std::unique_ptr<Type> type,
           std::vector<std::string> capture_names,
           std::vector<std::unique_ptr<Type>> capture_types,
           std::vector<size_t> capture_freevars,
           std::vector<std::string> arg_names,
           std::vector<DimVar> dimvars,
           std::vector<std::unique_ptr<ASTNode>> body)
          : ASTNode(std::move(dbg)),
            name(name),
            type(std::move(type)),
            capture_names(capture_names),
            capture_types(std::move(capture_types)),
            capture_freevars(capture_freevars),
            arg_names(arg_names),
            dimvars(dimvars),
            body(std::move(body)),
            // Below are initialized by inferDimvars() and compile() et al.
            // But set them here to avoid UB
            unique_gen_id(),
            capture_objs(),
            capture_instances(),
            dimvar_values(),
            needs_explicit_dimvars(false),
            missing_dimvars(),
            funcOp_name(),
            funcOp_private(false),
            funcOp(),
            capture_hashes() {}

    // Run the AST pass pipeline with a typechecker V. (The V chosen differs
    // between @qpu and @classical kernels.)
    template<typename V>
    void _runASTVisitorPipeline() {
        // Expand generic types and check for type variables we could not infer
        // based on captures
        {
            EvalDimVarExprVisitor evalVisitor(dimvar_values);
            walk(evalVisitor);
        }
        // Desugar -- want this to be distinct from canonicalization
        // since we seem to want to type-check after canonicalizing.
        // This limits which transformations can avoid typechecking.
        /*
        {
            DesugarVisitor desugarVisitor;
            walk(desugarVisitor);
        }
        */
        // Type check #1
        {
            V typeCheckVisitor;
            walk(typeCheckVisitor);
        }
        runExtraTypeChecking();
        // Canonicalize while we're at it
        {
            CanonicalizeVisitor canonVisitor;
            walk(canonVisitor);
        }
        // Type check #2, just in case canonicalize broke something
        {
            V typeCheckVisitor;
            walk(typeCheckVisitor);
        }
        runExtraTypeChecking();
    }

    // Extra per-subclass type checking
    virtual void runExtraTypeChecking() {}
    // Should run _runASTVisitorPipeline() above with the desired type checker V.
    virtual void runASTVisitorPipeline() = 0;
    virtual ASTKind kind() const = 0;
    void erase();
    bool needsRecompile(std::vector<HybridObj *> &provided_captures) const;
    void inferDimvarsFromCaptures(std::vector<std::unique_ptr<HybridObj>> provided_capture_objs);
    void registerExplicitDimvars(std::vector<std::optional<DimVarValue>> &values);
    // Used to infer type dimension variables of any captures kernels (if
    // possible)
    void inferCalleeDimvars();
    // Find and keep track of all instantations of captures, e.g.,
    // my_capture[[37]].
    void findCaptureInstantiations();
    // Run the AST visitor pipeline and then infer callee dimension variables
    // and synthesize instantiated captures
    virtual void typeCheck();
    // Meant to be overloaded by specific types of kernels. The parent class
    // here just calls compileIfNeeded() on every capture.
    virtual void compile(MlirHandle &handle);
    // If possible, call into the JIT'd code generated from this kernel
    virtual std::unique_ptr<KernelResult> call(MlirHandle &handle,
                                               std::string accelerator,
                                               size_t shots) = 0;
    // Return this kernel as OpenQASM 3.0
    virtual std::string qasm(MlirHandle &handle, bool print_locs) = 0;

    // Bit of a hack for copying: we need to copy over compile()-initialized state too
    void copyCompilerStateFrom(const Kernel &kernel) {
        capture_objs.clear();
        capture_objs.reserve(kernel.capture_objs.size());
        for (size_t i = 0; i < kernel.capture_objs.size(); i++) {
            capture_objs.push_back(std::move(kernel.capture_objs[i]->copyHybrid()));
        }
        capture_instances.clear();
        capture_instances.reserve(kernel.capture_instances.size());
        for (size_t i = 0; i < kernel.capture_instances.size(); i++) {
            const std::map<const std::vector<DimVarValue>, std::unique_ptr<HybridObj>> &theirs = *kernel.capture_instances[i];
            std::unique_ptr<std::map<const std::vector<DimVarValue>, std::unique_ptr<HybridObj>>> ours =
                    std::make_unique<std::map<const std::vector<DimVarValue>, std::unique_ptr<HybridObj>>>();
            for (const auto &p : theirs) {
                ours->emplace(p.first, std::move(p.second->copyHybrid()));
            }
            capture_instances.push_back(std::move(ours));
        }

        unique_gen_id = kernel.unique_gen_id;
        explicit_dimvars = kernel.explicit_dimvars;
        dimvar_values = kernel.dimvar_values;
        needs_explicit_dimvars = kernel.needs_explicit_dimvars;
        missing_dimvars = kernel.missing_dimvars;
        funcOp_name = kernel.funcOp_name;
        funcOp_private = kernel.funcOp_private;
        funcOp = kernel.funcOp;
        capture_hashes = kernel.capture_hashes;
    }
    // Return the name that was/will be used for the MLIR FuncOp for this kernel
    std::string getFuncOpName() const {
        if (funcOp_name.empty()) {
            assert(!unique_gen_id.empty()
                   && "Unique generation ID not set, bug in Python frontend?");
            return unique_gen_id;
        } else {
            return funcOp_name;
        }
    }
    virtual bool hasType() const override { return true; }
    virtual const Type &getType() const override {
        return *type;
    }
    // Helper to check the hashes of instantiated captures for equality
    static bool compareCaptureInstantiates(const CaptureInstantiates &lhs,
                                           const CaptureInstantiates &rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (size_t i = 0; i < lhs.size(); i++) {
            for (const auto &[dvvs, hybrid] : *lhs[i]) {
                if (!rhs[i]->count(dvvs)
                        || hybrid->getHash() != (*rhs[i])[dvvs]->getHash()) {
                    return false;
                }
            }
            for (const auto &[dvvs, hybrid] : *rhs[i]) {
                if (!lhs[i]->count(dvvs)
                        || hybrid->getHash() != (*lhs[i])[dvvs]->getHash()) {
                    return false;
                }
            }
        }
        return true;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const Kernel &kern = static_cast<const Kernel &>(node);
        return name == kern.name
               && capture_names == kern.capture_names
               && compare_vectors(capture_types, kern.capture_types)
               && capture_freevars == kern.capture_freevars
               && arg_names == kern.arg_names
               && dimvars == kern.dimvars
               && compare_vectors(body, kern.body)
               && unique_gen_id == kern.unique_gen_id
               && explicit_dimvars == kern.explicit_dimvars
               && compare_vectors_of_hashable(capture_objs, kern.capture_objs)
               && compareCaptureInstantiates(capture_instances, kern.capture_instances)
               && dimvar_values == kern.dimvar_values
               && needs_explicit_dimvars == kern.needs_explicit_dimvars
               && missing_dimvars == kern.missing_dimvars
               && funcOp_name == kern.funcOp_name
               && funcOp_private == kern.funcOp_private
               && funcOp == kern.funcOp
               && capture_hashes == kern.capture_hashes;
    }
    virtual std::unique_ptr<HybridObj> copyHybrid() const override {
        std::unique_ptr<ASTNode> copied = copy();
        return unique_downcast<Kernel, ASTNode>(copied);
    }
    virtual HybridObj::Hash getHash() const override { return "Kernel:" + unique_gen_id; }
    virtual bool needsExplicitDimvars() const override { return needs_explicit_dimvars; }
    virtual void getMissingDimvars(std::vector<DimVar> &missing_dimvars_out) const override {
        missing_dimvars_out = missing_dimvars;
    }
    virtual void evalExplicitDimvars(DimVarValues &vals, size_t n_freevars) override;
    virtual void instantiateWithExplicitDimvars(std::vector<DimVarValue> &values) override {
        std::vector<std::optional<DimVarValue>> vals;
        vals.reserve(values.size());
        // No more free variables, require everything set
        for (auto val : values) {
            vals.emplace_back(val);
        }
        registerExplicitDimvars(vals);
        typeCheck();
        // TODO: generate a graphviz AST here too. ideally, put that code in
        //       C++ and then let python call it
    }
    virtual bool isCompileNeeded() const {
        return !funcOp;
    }
    virtual void compileIfNeeded(MlirHandle &handle,
                                 std::string funcOp_name_,
                                 bool funcOp_private_) override {
        if (isCompileNeeded()) {
            funcOp_name = funcOp_name_;
            funcOp_private = funcOp_private_;
            compile(handle);
        }
    }
    virtual void eraseIfPrivate() override {
        if (funcOp_private) {
            erase();
        }
    }
    virtual std::string label() override { return "Kernel"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override;
    VISITOR_TEDIUM
};

// The subclass of Kernel specifically for @qpu kernels.
struct QpuKernel : Kernel {
    QpuKernel(std::unique_ptr<DebugInfo> dbg,
           std::string name,
           std::unique_ptr<Type> type,
           std::vector<std::string> capture_names,
           std::vector<std::unique_ptr<Type>> capture_types,
           std::vector<size_t> capture_freevars,
           std::vector<std::string> arg_names,
           std::vector<DimVar> dimvars,
           std::vector<std::unique_ptr<ASTNode>> body)
          : Kernel(std::move(dbg),
                   name,
                   std::move(type),
                   capture_names,
                   std::move(capture_types),
                   capture_freevars,
                   arg_names,
                   dimvars,
                   std::move(body)) {}

    virtual ASTKind kind() const override { return AST_QPU; }
    virtual void compile(MlirHandle &handle) override;
    virtual std::unique_ptr<KernelResult> call(MlirHandle &handle,
                                               std::string accelerator,
                                               size_t shots) override;
    // See docs/qiree.md for more information on QIR-EE integration.
#ifdef QWERTY_USE_QIREE
    // Run through XACC via QIR-EE
    virtual std::unique_ptr<KernelResult> callQiree(
        MlirHandle &handle,
        std::string accelerator,
        size_t n_shots,
        const BitType *output_type);
#endif
    // Simulate using the QIR simulation runtime bundled from qir-runner
    virtual std::unique_ptr<KernelResult> callLocalSim(
        MlirHandle &handle,
        size_t n_shots,
        const BitType *output_type);
    virtual std::string qasm(MlirHandle &handle, bool print_locs) override;
    // The extra type checking here is just checking that bases are not passed
    // around at runtime, which would be invalid.
    virtual void runExtraTypeChecking() override {
        FlagImmaterialVisitor dynBasisVisitor;
        walk(dynBasisVisitor);
    }
    virtual void runASTVisitorPipeline() override {
        // Here?
        _runASTVisitorPipeline<QpuTypeCheckVisitor>();
    }
    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<QpuKernel> copied = std::make_unique<QpuKernel>(
            std::move(dbg->copy()),
            name,
            std::move(type->copy()),
            capture_names,
            std::move(copy_vector_of_copyable<Type>(capture_types)),
            capture_freevars,
            arg_names,
            dimvars,
            std::move(copy_vector_of_copyable<ASTNode>(body)));
        copied->copyInternalsFrom(*this);
        copied->copyCompilerStateFrom(*this);
        return std::move(copied);
    }
};

// The subclass of Kernel specifically for @classical kernels.
struct ClassicalKernel : Kernel {
    std::vector<uint32_t> output_wires;
    std::unordered_map<EmbeddingKind, qwerty::FuncOp> embedding_func_ops;

    ClassicalKernel(std::unique_ptr<DebugInfo> dbg,
                    std::string name,
                    std::unique_ptr<Type> type,
                    std::vector<std::string> capture_names,
                    std::vector<std::unique_ptr<Type>> capture_types,
                    std::vector<size_t> capture_freevars,
                    std::vector<std::string> arg_names,
                    std::vector<DimVar> dimvars,
                    std::vector<std::unique_ptr<ASTNode>> body)
                   : Kernel(std::move(dbg),
                            name,
                            std::move(type),
                            capture_names,
                            std::move(capture_types),
                            capture_freevars,
                            arg_names,
                            dimvars,
                            std::move(body)) {}

    virtual ASTKind kind() const override { return AST_CLASSICAL; }
    using Kernel::getFuncOpName; // https://stackoverflow.com/a/1734902/321301
    std::string getFuncOpName(EmbeddingKind embedding) const {
        return getFuncOpName() + "__" + embedding_kind_name(embedding);
    }
    // Synthesize a FuncOp for some embedding (sign, xor, etc)
    qwerty::FuncOp getFuncOp(MlirHandle &handle, ClassicalKernel *operand,
                             EmbeddingKind embedding);
    virtual bool isCompileNeeded() const override {
        return embedding_func_ops.empty();
    }
    virtual void typeCheck() override;
    virtual void compile(MlirHandle &handle) override;
    virtual std::unique_ptr<KernelResult> call(MlirHandle &handle,
                                               std::string accelerator,
                                               size_t shots) override {
        throw JITException(ast_kind_name(kind()) + " kernels cannot be directly "
                           "invoked right now");
    }
    virtual std::string qasm(MlirHandle &handle, bool print_locs) override {
        throw JITException(ast_kind_name(kind()) + " kernels cannot be "
                           "converted to QASM right now");
    }
    virtual void runASTVisitorPipeline() override {
        _runASTVisitorPipeline<ClassicalTypeCheckVisitor>();
    }
    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ClassicalKernel> copied = std::make_unique<ClassicalKernel>(
            std::move(dbg->copy()),
            name,
            std::move(type->copy()),
            capture_names,
            std::move(copy_vector_of_copyable<Type>(capture_types)),
            capture_freevars,
            arg_names,
            dimvars,
            std::move(copy_vector_of_copyable<ASTNode>(body)));
        copied->copyInternalsFrom(*this);
        copied->copyCompilerStateFrom(*this);
        // Copy ClassicalKernel-specific state too
        copied->output_wires = output_wires;
        copied->embedding_func_ops = embedding_func_ops;
        return std::move(copied);
    }
};

// @classical nodes

// A unary operation on bits in a @classical kernel.
// Example syntax:
//     ~x
struct BitUnaryOp : Expr {
    BitOp op;
    std::unique_ptr<ASTNode> operand;
    std::unique_ptr<Type> type;

    BitUnaryOp(std::unique_ptr<DebugInfo> dbg,
               BitOp op,
               std::unique_ptr<ASTNode> operand)
              : Expr(std::move(dbg)),
                op(op),
                operand(std::move(operand)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BitUnaryOp>(std::move(dbg->copy()),
                                             op,
                                             std::move(operand->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BitUnaryOp &unop = static_cast<const BitUnaryOp &>(node);
        return op == unop.op
               && *operand == *unop.operand;
    }
    virtual std::string label() override { return "BitUnaryOp"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"operand", operand}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Operation: " + bit_op_name(op)};
    }
    VISITOR_TEDIUM
};

// A binary operation on bits in a @classical kernel.
// Example syntax:
//     x & y
struct BitBinaryOp : Expr {
    BitOp op;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    std::unique_ptr<Type> type;

    BitBinaryOp(std::unique_ptr<DebugInfo> dbg,
                BitOp op,
                std::unique_ptr<ASTNode> left,
                std::unique_ptr<ASTNode> right)
               : Expr(std::move(dbg)),
                 op(op),
                 left(std::move(left)),
                 right(std::move(right)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BitBinaryOp>(std::move(dbg->copy()),
                                              op,
                                              std::move(left->copy()),
                                              std::move(right->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BitBinaryOp &binop = static_cast<const BitBinaryOp &>(node);
        return op == binop.op
               && *left == *binop.left
               && *right == *binop.right;
    }
    virtual std::string label() override { return "BitBinaryOp"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"left", left}, {"right", right}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Operation: " + bit_op_name(op)};
    }
    VISITOR_TEDIUM
};

// A reduction operation on bits in a @classical kernel.
// Example syntax:
//     x.xor_reduce()
struct BitReduceOp : Expr {
    BitOp op;
    std::unique_ptr<ASTNode> operand;
    std::unique_ptr<Type> type;

    BitReduceOp(std::unique_ptr<DebugInfo> dbg,
                BitOp op,
                std::unique_ptr<ASTNode> operand)
               : Expr(std::move(dbg)),
                 op(op),
                 operand(std::move(operand)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BitReduceOp>(std::move(dbg->copy()),
                                              op,
                                              std::move(operand->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BitReduceOp &redop = static_cast<const BitReduceOp &>(node);
        return op == redop.op
               && *operand == *redop.operand;
    }
    virtual std::string label() override { return "BitReduceOp"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"operand", operand}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Operation: " + bit_op_name(op)};
    }
    VISITOR_TEDIUM
};

// Concatenate bits in a @classical kernel
// Example syntax:
//     (x, y)
struct BitConcat : Expr {
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;
    std::unique_ptr<Type> type;

    BitConcat(std::unique_ptr<DebugInfo> dbg,
              std::unique_ptr<ASTNode> left,
              std::unique_ptr<ASTNode> right)
             : Expr(std::move(dbg)),
               left(std::move(left)),
               right(std::move(right)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BitConcat>(std::move(dbg->copy()),
                                            std::move(left->copy()),
                                            std::move(right->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BitConcat &concat = static_cast<const BitConcat &>(node);
        return *left == *concat.left
               && *right == *concat.right;
    }
    virtual std::string label() override { return "BitConcat"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"left", left}, {"right", right}};
    }
    VISITOR_TEDIUM
};

// Repeatedly concate a bit[N] with itself.
// Example syntax:
//     bit[1](0b0).repeat(5)
// This example is equivalent to bit[5](0b00000).
struct BitRepeat : Expr {
    std::unique_ptr<ASTNode> bits;
    std::unique_ptr<DimVarExpr> amt;
    std::unique_ptr<Type> type;

    BitRepeat(std::unique_ptr<DebugInfo> dbg,
              std::unique_ptr<ASTNode> bits,
              std::unique_ptr<DimVarExpr> amt)
             : Expr(std::move(dbg)),
               bits(std::move(bits)),
               amt(std::move(amt)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BitRepeat>(std::move(dbg->copy()),
                                            std::move(bits->copy()),
                                            std::move(amt->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BitRepeat &other = static_cast<const BitRepeat &>(node);
        return *bits == *other.bits
               && *amt == *other.amt;
    }
    virtual std::string label() override { return "BitRepeat"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"bits", bits}};
    }
    virtual std::vector<std::string> getAdditionalMetadata() const override {
        return {"Amount: " + amt->toString()};
    }
    VISITOR_TEDIUM
};

// Modular multiplier for @classical kernels.
// Example syntax:
//     x**2**j*y % modN
struct ModMulOp : Expr {
    std::unique_ptr<DimVarExpr> x;
    std::unique_ptr<DimVarExpr> j;
    std::unique_ptr<ASTNode> y;
    std::unique_ptr<DimVarExpr> modN;
    std::unique_ptr<Type> type;

    ModMulOp(std::unique_ptr<DebugInfo> dbg,
              std::unique_ptr<DimVarExpr> x,
              std::unique_ptr<DimVarExpr> j,
              std::unique_ptr<ASTNode> y,
              std::unique_ptr<DimVarExpr> modN)
            : Expr(std::move(dbg)),
              x(std::move(x)),
              j(std::move(j)),
              y(std::move(y)),
              modN(std::move(modN)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<ModMulOp>(std::move(dbg->copy()),
                                           std::move(x->copy()),
                                           std::move(j->copy()),
                                           std::move(y->copy()),
                                           std::move(modN->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const ModMulOp &modmul = static_cast<const ModMulOp &>(node);
        return *x == *modmul.x
               && *j == *modmul.j
               && *y == *modmul.y
               && *modN == *modmul.modN;
    }
    virtual std::string label() override { return "ModMulOp"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {{"y", y}};
    }
    VISITOR_TEDIUM
};

// A bit[N] literal.
// Example syntax:
//     bit[4](0b1101)
struct BitLiteral : Expr {
    std::unique_ptr<DimVarExpr> val;
    std::unique_ptr<DimVarExpr> n_bits;
    std::unique_ptr<Type> type;

    BitLiteral(std::unique_ptr<DebugInfo> dbg,
               std::unique_ptr<DimVarExpr> val,
               std::unique_ptr<DimVarExpr> n_bits)
              : Expr(std::move(dbg)),
                val(std::move(val)),
                n_bits(std::move(n_bits)) {}

    virtual std::unique_ptr<ASTNode> copy() const override {
        std::unique_ptr<ASTNode> result =
                std::make_unique<BitLiteral>(std::move(dbg->copy()),
                                             std::move(val->copy()),
                                             std::move(n_bits->copy()));
        result->copyInternalsFrom(*this);
        return std::move(result);
    }
    virtual bool hasType() const override { return !!type; }
    virtual const Type &getType() const override {
        return *type;
    }
    virtual bool isEqual(const ASTNode &node) const override {
        const BitLiteral &bitlit = static_cast<const BitLiteral &>(node);
        return *val == *bitlit.val
               && *n_bits == *bitlit.n_bits;
    }
    virtual std::string label() override { return "BitLiteral"; }
    virtual std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> children() override {
        return {};
    }
    VISITOR_TEDIUM
};

#endif
