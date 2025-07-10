//===- QwertyOps.td - Qwerty dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_OPS_TD
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_OPS_TD

include "QCirc/IR/QCircTypes.td"
include "QCirc/IR/QCircInterfaces.td"
include "Qwerty/IR/QwertyDialect.td"
include "Qwerty/IR/QwertyTypes.td"
include "Qwerty/IR/QwertyAttributes.td"
include "Qwerty/IR/QwertyInterfaces.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Based heavily on the Func dialect's FuncOp
def Qwerty_FuncOp : Qwerty_Op<"func", [
        AffineScope, AutomaticAllocationScope, IsolatedFromAbove,
        DeclareOpInterfaceMethods<CallableOpInterface>,
        DeclareOpInterfaceMethods<FunctionOpInterface, ["cloneTypeWith", "verifyBody"]>]> {
    let summary = "A qwerty function (kernel)";
    let description = [{
        Unlike `func.func`, a `qwerty.func` is guaranteed to always have a
        body. It also has captures, which are specified with additional block
        arguments that precede the block arguments corresponding to the
        function type. That is, the FunctionType does not include capture
        types — you have to find them from the entry block arguments (see
        `getNumCaptures()` and `getCapturesTypes()`).
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<Qwerty_FunctionType>:$qwerty_func_type,
                         OptionalAttr<StrAttr>:$sym_visibility,
                         OptionalAttr<DictArrayAttr>:$arg_attrs,
                         OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$sym_name, "FunctionType":$qwerty_func_type), [{
            build($_builder, $_state, sym_name, qwerty_func_type, nullptr, nullptr, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
        unsigned getNumCaptures() {
            return getBody().getNumArguments()
                   - getQwertyFuncType().getFunctionType().getInputs().size();
        };

        void getCaptureTypes(llvm::SmallVectorImpl<mlir::Type> &capture_types_out) {
            capture_types_out.clear();
            auto arg_types = getBody().getArgumentTypes();
            capture_types_out.append(
                arg_types.begin(),
                arg_types.begin() + getNumCaptures());
        }
    }];
}

def Qwerty_CallOp : Qwerty_Op<"call", [
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Calls a qwerty function";
    let description = [{
        Unlike `func.call`, a `qwerty.call` can call a function in a predicated
        or adjointed form. Captures may also be passed, but they are not
        distinguished from arguments here because without consulting the symbol
        table, we do not have enough information to know where the captures end
        and the operands begin. That's fine — the arguments here will still map
        1:1 with the block arguments of the callee.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee,
                     BoolAttr:$adj,
                     OptionalAttr<Qwerty_BasisAttr>:$pred,
                     Variadic<AnyType>:$captures_and_operands);
    let results = (outs Variadic<AnyType>:$results);

    let builders = [
        OpBuilder<(ins "qwerty::FuncOp":$callee, CArg<"::mlir::ValueRange", "{}">:$captures_and_operands), [{
            mlir::TypeRange results =
                callee.getQwertyFuncType().getFunctionType().getResults();
            build($_builder, $_state, results, callee.getSymName(), /*adj=*/false, /*pred=*/nullptr, captures_and_operands);
        }]>,
        OpBuilder<(ins "::mlir::TypeRange":$result_types, "::mlir::FlatSymbolRefAttr":$callee, CArg<"mlir::ValueRange", "{}">:$captures_and_operands), [{
            build($_builder, $_state, result_types, callee, /*adj=*/false, /*pred=*/nullptr, captures_and_operands);
        }]>,
        OpBuilder<(ins "qwerty::FuncOp":$callee, "bool":$adj, "qwerty::BasisAttr":$pred, CArg<"::mlir::ValueRange", "{}">:$captures_and_operands), [{
            mlir::TypeRange results =
                callee.getQwertyFuncType().getFunctionType().getResults();
            if (pred) {
                assert(results.size() == 1 && llvm::isa<qwerty::QBundleType>(results[0]));
                qwerty::QBundleType res_type = llvm::cast<qwerty::QBundleType>(results[0]);
                results = $_builder.getType<qwerty::QBundleType>(res_type.getDim() + pred.getDim());
            }
            build($_builder, $_state, results, callee.getSymName(), adj, pred, captures_and_operands);
        }]>,
        OpBuilder<(ins "::mlir::TypeRange":$result_types, "::mlir::FlatSymbolRefAttr":$callee, "bool":$adj, "qwerty::BasisAttr":$pred, CArg<"mlir::ValueRange", "{}">:$captures_and_operands), [{
            build($_builder, $_state, result_types, callee, $_builder.getBoolAttr(adj), pred, captures_and_operands);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;
}

def Qwerty_FuncConstOp : Qwerty_Op<"func_const", [
        Pure, Stationary,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Generates a function value given the symbol name of a qwerty.func";
    let description = [{
        Unlike `func.func_const`, this op requires a list of captures.
    }];

    let arguments = (ins FlatSymbolRefAttr:$func, Variadic<AnyType>:$captures);
    let results = (outs Qwerty_FunctionType:$result);

    let builders = [
        OpBuilder<(ins "qwerty::FuncOp":$func, CArg<"::mlir::ValueRange", "{}">:$captures), [{
            build($_builder, $_state, func.getQwertyFuncType(), func.getSymName(), captures);
        }]>
    ];

    let assemblyFormat = "$func `[` $captures `]` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_FuncAdjointOp : Qwerty_Op<"func_adj", [
        InferTypeOpAdaptor, Pure, Stationary]> {
    let summary = "Take the adjoint of a Qwerty function";

    let arguments = (ins Qwerty_FunctionType:$callee);
    let results = (outs Qwerty_FunctionType:$result);

    let hasCanonicalizer = 1;
    let hasVerifier = 1;

    let assemblyFormat = "$callee attr-dict `:` functional-type(operands, results)";
}

def Qwerty_FuncPredOp : Qwerty_Op<"func_pred", [
        InferTypeOpAdaptor, Pure, Stationary]> {
    let summary = "Predicate a Qwerty function";
    let description = [{
        The result of this op is the same as the input except the input and
        output qbundle is extended by the dimension of the basis operand
        (`pred`).
    }];

    let arguments = (ins Qwerty_BasisAttr:$pred, Qwerty_FunctionType:$callee);
    let results = (outs Qwerty_FunctionType:$result);

    let hasCanonicalizer = 1;
    let hasVerifier = 1;

    let assemblyFormat = "$callee `by` $pred attr-dict `:` functional-type(operands, results)";
}

def Qwerty_CallIndirectOp : Qwerty_Op<"call_indirect", [
        InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<CallOpInterface>,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Call a qwerty function pointer";

    let arguments = (ins Qwerty_FunctionType:$callee,
                         Variadic<AnyType>:$call_operands);
    let results = (outs Variadic<AnyType>:$results);

    let hasCanonicalizer = 1;

    let assemblyFormat = "$callee `(` $call_operands `)` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_LambdaOp : Qwerty_Op<"lambda", [
        AutomaticAllocationScope, AffineScope, IsolatedFromAbove, Pure,
        Stationary, DeclareOpInterfaceMethods<CallableOpInterface>]> {
    let description = [{
        A cross between a `qwerty.func` and a `qwerty.func_const`. This op does
        not have a symbol name; instead, it yields a function value directly.
        Also like `qwerty.func_const`, it takes a list of capture values.
        As with a `qwerty.func`, the FunctionType (in this case the result
        type) does not include capture types, but those capture types are the
        leading block arguments.
    }];
    let arguments = (ins Variadic<AnyType>:$captures);
    let results = (outs Qwerty_FunctionType:$result);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
    // TODO: add verifier to sanity check things

    let extraClassDeclaration = [{
        unsigned getNumCaptures() {
            return getRegion().getNumArguments()
                   - getResult().getType().getFunctionType().getInputs().size();
        };
    }];
}

def Qwerty_ReturnOp : Qwerty_Op<"return", [Pure, ParentOneOf<["FuncOp", "LambdaOp"]>,
                                           ReturnLike, Terminator]> {
    let summary = "Terminator for qwerty.func and qwerty.lambda";
    let arguments = (ins Variadic<AnyType>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` qualified(type($operands)))?";
}

def Qwerty_PredOp : Qwerty_Op<"pred", [
        DeclareOpInterfaceMethods<RegionBranchOpInterface>,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Region to predicate in-place";
    let description = [{
        Contains a region of code that should be predicated. This is
        currently neither emitted by the frontend nor inserted by any pass. It
        remains only to test `PredBlock.cpp` via `InlinePredPass.cpp`.
        (That is, the block held inside this op is predicated the same way as
        a predicated function body would be.)
    }];
    let arguments = (ins Qwerty_BasisAttr:$basis, Qwerty_QBundle:$predBundleIn, Qwerty_QBundle:$regionArg);
    let results = (outs Qwerty_QBundle:$predBundleOut, Qwerty_QBundle:$regionResult);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

// This is intentionally NOT return-like. Otherwise, MLIR gets very confused by
// how PredOp works
def Qwerty_YieldOp : Qwerty_Op<"yield", [Pure, HasParent<"PredOp">, Terminator]> {
    let summary = "Terminator for qwerty.pred";
    let arguments = (ins Qwerty_QBundle:$qbundle);
    let assemblyFormat = "$qbundle attr-dict `:` qualified(type($qbundle))";
}

def Qwerty_SuperposOp: Qwerty_Op<"superpos", [InferTypeOpAdaptor]> {
    let summary = "Prepare a superposition state";
    let arguments = (ins Qwerty_SuperposAttr:$superpos);
    let results = (outs Res<Qwerty_QBundle, "", [MemAlloc]>:$qbundle);
    let hasCanonicalizer = 1;
    let assemblyFormat = "$superpos attr-dict `:` qualified(type($qbundle))";
}

//////// QBUNDLE MANAGEMENT ////////

def Qwerty_QBundlePrepOp : Qwerty_Op<"qbprep", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable, ["isAdjointable", "getNumOperandsOfAdjoint"]>,
        DeclareOpInterfaceMethods<Predicatable>]> {
    let summary = "Prepare qubits in the specified eigenstate";
    let description = [{
        Prepares `dim` qubits in the specified `eigenstate` of the provided
        primitive basis.
    }];
    let arguments = (ins Qwerty_PrimitiveBasisAttr:$prim_basis, Qwerty_EigenstateAttr:$eigenstate, I64Attr:$dim);
    let results = (outs Res<Qwerty_QBundle, "", [MemAlloc]>:$result);
    let hasVerifier = 1;
    let assemblyFormat = "attr-dict $prim_basis `<` $eigenstate `>` `` `[` $dim `]` `:` functional-type(operands, results)";
}

def Qwerty_QBundlePackOp : Qwerty_Op<"qbpack", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable, ["getNumOperandsOfAdjoint"]>,
        DeclareOpInterfaceMethods<Predicatable>]> {
    let summary = "Combine N qubits into a qbundle[N]";
    let arguments = (ins Variadic<QCirc_Qubit>:$qubits);
    let results = (outs Qwerty_QBundle:$qbundle);
    let hasVerifier = 1;
    let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundleUnpackOp : Qwerty_Op<"qbunpack", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable, ["getNumOperandsOfAdjoint"]>,
        DeclareOpInterfaceMethods<Predicatable>]> {
    let summary = "Split a qbundle[N] into N qubits";
    let arguments = (ins Qwerty_QBundle:$qbundle);
    let results = (outs Variadic<QCirc_Qubit>:$qubits);
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
    let assemblyFormat = "$qbundle attr-dict `:` functional-type(operands, results)";
}

//////// INITIALIZING QUBITS ////////

def Qwerty_BitInitOp : Qwerty_Op<"bitinit", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>]> {
    let summary = "Prepare qubit state '1100' given a bitbundle operand containing 0b1100";
    let arguments = (ins Qwerty_QBundle:$qbundle);
    let arguments = (ins Qwerty_BitBundle:$bitBundle, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `as` $bitBundle attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundleInitOp : Qwerty_Op<"qbinit", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Prepare singleton basis given zeros as input";
    let arguments = (ins Qwerty_BasisAttr:$basis, Variadic<F64>:$basisPhases, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `as` $basis (`phases` ` ` `(` $basisPhases^ `)`)? attr-dict `:` functional-type(operands, results)";
}

// Undo a qbinit
def Qwerty_QBundleDeinitOp : Qwerty_Op<"qbdeinit", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Undo a qwerty.qbinit";
    let arguments = (ins Qwerty_BasisAttr:$basis, Variadic<F64>:$basisPhases, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `as` $basis (`phases` ` ` `(` $basisPhases^ `)`)? attr-dict `:` functional-type(operands, results)";
}

//////// QBUNDLE OPERATIONS ////////

// This is intentionally not marked with the Pure trait: we want to explicitly
// discard qubits (aka qfree in a lower-level dialect)
def Qwerty_QBundleDiscardOp : Qwerty_Op<"qbdiscard"> {
    let summary = "Reset and free a qbundle";
    let arguments = (ins Arg<Qwerty_QBundle, "", [MemFree]>:$qbundle);
    let assemblyFormat = "$qbundle attr-dict `:` functional-type(operands, results)";
}

// The input qbundle MUST be zero or horrific things will happen
def Qwerty_QBundleDiscardZeroOp : Qwerty_Op<"qbdiscardz", [
        DeclareOpInterfaceMethods<Adjointable, ["getNumOperandsOfAdjoint"]>,
        DeclareOpInterfaceMethods<Predicatable>]> {
    let summary = "Free a qbundle without resetting, assuming it is in the zero state";
    let arguments = (ins Arg<Qwerty_QBundle, "", [MemFree]>:$qbundle);
    let assemblyFormat = "$qbundle attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundleIdentityOp : Qwerty_Op<"qbid", [
        Pure, SameOperandsAndResultType,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable>]> {
    let summary = "Identity operation";
    let arguments = (ins Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundlePhaseOp : Qwerty_Op<"qbphase", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Impart a global phase of e^{i*theta} on the provided qbundle";
    let arguments = (ins F64:$theta, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `by` `exp` `(` `i` `` `*` `` $theta `)` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundleBasisTranslationOp : Qwerty_Op<"qbtrans", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Perform a basis translation";
    let description = [{
        The `basisPhases` list is the concatenation of a list for the input
        basis and another list for the output basis.
    }];
    let arguments = (ins Qwerty_BasisAttr:$basisIn, Qwerty_BasisAttr:$basisOut, Variadic<F64>:$basisPhases, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `by` $basisIn ` ` `>` `>` $basisOut (`phases` ` ` `(` $basisPhases^ `)`)? attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundleMeasureOp : Qwerty_Op<"qbmeas", [Pure, InferTypeOpAdaptor]> {
    let summary = "Measure a qbundle, yielding a bitbundle";
    let arguments = (ins Qwerty_BasisAttr:$basis, Qwerty_QBundle:$qbundle);
    let results = (outs Qwerty_BitBundle:$bits);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundle `by` $basis attr-dict `:` qualified(type($qbundle)) `->` qualified(type($bits))";
}

def Qwerty_QBundleProjectOp : Qwerty_Op<"qbproj", [Pure, InferTypeOpAdaptor]> {
    let summary = "Measure a qbundle, yielding the projected qbundle";
    let arguments = (ins Qwerty_BasisAttr:$basis, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `by` $basis attr-dict `:` qualified(type($qbundleIn)) `->` qualified(type($qbundleOut))";
}

def Qwerty_QBundleFlipOp : Qwerty_Op<"qbflip", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Apply translation that swaps basis vectors";
    let arguments = (ins Qwerty_BasisAttr:$basis, Variadic<F64>:$basisPhases, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `by` $basis `with` ` ` `(` $basisPhases `)` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_QBundleRotateOp : Qwerty_Op<"qbrot", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Predicatable, ["buildPredicated"]>]> {
    let summary = "Apply the .rotate Qwerty primitive to a qbundle";
    let arguments = (ins Qwerty_BasisAttr:$basis, F64:$theta, Qwerty_QBundle:$qbundleIn);
    let results = (outs Qwerty_QBundle:$qbundleOut);
    let hasVerifier = 1;
    let assemblyFormat = "$qbundleIn `by` $basis `by` $theta attr-dict `:` functional-type(operands, results)";
}

//////// BIT BUNDLE OPERATIONS ////////

def Qwerty_BitBundlePackOp : Qwerty_Op<"bitpack", [
        InferTypeOpAdaptor, Stationary]> {
    let summary = "Combine N i1s into a bitbundle[N]";
    let arguments = (ins Variadic<I1>:$bits);
    let results = (outs Res<Qwerty_BitBundle, "", [MemAlloc]>:$bundle);
    let hasVerifier = 1;
    let assemblyFormat = "`(` $bits `)` attr-dict `:` functional-type(operands, results)";
}

def Qwerty_BitBundleUnpackOp : Qwerty_Op<"bitunpack", [
        InferTypeOpAdaptor, Stationary]> {
    let summary = "Split a bitbundle[N] into N i1s";
    let arguments = (ins Arg<Qwerty_BitBundle, "", [MemFree]>:$bundle);
    let results = (outs Variadic<I1>:$bits);
    let hasCanonicalizer = 1;
    let assemblyFormat = "$bundle attr-dict `:` functional-type(operands, results)";
}

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_OPS_TD
