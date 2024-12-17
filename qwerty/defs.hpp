#ifndef DEFS_H
#define DEFS_H

#include <string>
#include <unordered_map>

// This header file has forward declarations to avoid a dependence cycle
// between ast.hpp and other headers

// ...and some weird one-off stuff like this guy
extern bool qwerty_debug;

#ifdef _MSC_VER
    #include <intrin.h>
    static unsigned long bits_trailing_zeros(uint64_t x) {
        unsigned long ret;
        _BitScanReverse64(&ret, x);
        return ret;
    }
    #define bits_needed(x) ((sizeof (unsigned __int64))*CHAR_BIT-__lzcnt64(x))
    #define bits_popcount(x) __popcnt64(x)
    // Keeping this around if needed if Windows trolls us further
    //#include <bitset>
    //static unsigned long bits_popcount(uint64_t x) {
    //    return std::bitset<64>(x).count();
    //}
#else
    #define bits_trailing_zeros(x) ((unsigned int)__builtin_ctzll(x))
    #define bits_needed(x) ((sizeof (unsigned long long))*CHAR_BIT-__builtin_clzll(x))
    #define bits_popcount(x) __builtin_popcountll(x)
#endif

// The order here MUST match the ordering in QwertyAttributes.td... or else!
// For information on what this is, please see QwertyAttributes.td.
typedef enum Eigenstate {
    PLUS,
    MINUS
} Eigenstate;

// The order here MUST match the ordering in QwertyAttributes.td... or else!
// For information on what this is, please see QwertyAttributes.td.
typedef enum PrimitiveBasis {
    X,
    Y,
    Z,
    FOURIER,
} PrimitiveBasis;

// Non-nodes
struct HybridObj;
struct Bits;
struct DebugInfo;
struct DimVarExpr;
// Shared nodes
struct QpuKernel;
struct ClassicalKernel;
struct Type;
struct FuncType;
struct BroadcastType;
struct ASTNode;
struct Variable;
struct Assign;
struct DestructAssign;
struct Return;
struct Kernel;
struct Slice;
// @qpu nodes
struct Adjoint;
struct Prepare;
struct Lift;
struct EmbedClassical;
struct Pipe;
struct Instantiate;
struct Repeat;
struct RepeatTensor;
struct Pred;
struct BiTensor;
struct BroadcastTensor;
struct QubitLiteral;
struct Phase;
struct FloatLiteral;
struct FloatNeg;
struct FloatBinaryOp;
struct FloatDimVarExpr;
struct TupleLiteral;
struct BuiltinBasis;
struct Identity;
struct BasisTranslation;
struct Discard;
struct Measure;
struct Project;
struct Flip;
struct Rotate;
struct BasisLiteral;
struct SuperposLiteral;
struct Conditional;
// @classical nodes
struct BitUnaryOp;
struct BitBinaryOp;
struct BitReduceOp;
struct BitConcat;
struct BitRepeat;
struct ModMulOp;
struct BitLiteral;

// https://stackoverflow.com/a/35368387/321301
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

using DimVar = std::string;
using DimVarValue = ssize_t;
using DimVarValues = std::unordered_map<DimVar, DimVarValue>;

#endif // DEFS_H
