//===- QCircOps.td - QCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_OPS_TD
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_OPS_TD

include "QCirc/IR/QCircDialect.td"
include "QCirc/IR/QCircAttributes.td"
include "QCirc/IR/QCircInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def QCirc_CalcOp : QCirc_Op<"calc", [
        Pure,
        Stationary,
        IsolatedFromAbove,
        DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
    let summary = "A region that should be left alone when calculating the adjoint";
    let description = [{
        A chunk of "stationary" code that should not be reversed. Usually arith
        ops calculating some phase angle.

        Arguments are passed in via the inputs, through the block arguments,
        through the ops in the region, through the block terminator operand,
        and then placed in the outputs. The IsolatedFromAbove trait mandates
        this flow, dramatically simplifying canonicalizing chains of calcs
        into 1 calc.
    }];
    let arguments = (ins Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;
    let hasFolder = 1;
    // TODO: add verifier that all contained ops are Pure
}

def QCirc_CalcYieldOp : QCirc_Op<"calc_yield", [Pure, HasParent<"CalcOp">, Terminator]> {
    let summary = "Terminator for qcirc.calc";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "`(` $results `)` attr-dict `:` qualified(type(operands))";
}

def QCirc_AdjointOp : QCirc_Op<"adj", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
    let summary = "Region to adjoint in-place";
    let description = [{
        Contains a region of code that should be adjointed. This is
        currently neither emitted by the frontend nor inserted by any pass. It
        remains only to test `AdjointBlock.cpp` via `InlineAdjPass.cpp`.
        (That is, the block held inside this op is adjointed the same way as an
        adjointed function body would be.)
    }];
    let arguments = (ins Variadic<QCirc_Qubit>:$inputs);
    let results = (outs Variadic<QCirc_Qubit>:$results);
    let regions = (region SizedRegion<1>:$region);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

// This is intentionally NOT return-like. Otherwise, MLIR got very confused
def QCirc_YieldOp : QCirc_Op<"yield", [
        Pure, HasParent<"AdjointOp">, Terminator]> {
    let summary = "Terminator for qcirc.adj";
    let arguments = (ins Variadic<QCirc_Qubit>:$qubits);
    let assemblyFormat = "`(` $qubits `)` attr-dict `:` qualified(type(operands))";
}

def QCirc_Gate1QOp : QCirc_Op<"gate1q", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Controllable>]> {
    let summary = "A quantum gate with 1 target and no real-valued parameters";
    let description = [{
        Dataflow semantics one-qubit quantum gate with a variable number of
        control qubits. (That is, qubits flow through this gate like a typical
        SSA op.)
    }];
    let arguments = (ins QCirc_Gate1QAttr:$gate, Variadic<QCirc_Qubit>:$controls, QCirc_Qubit:$qubit);
    let results = (outs Variadic<QCirc_Qubit>:$controlResults, QCirc_Qubit:$result);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        static llvm::StringRef getGateAttrStrName() { return "gate"; }
    }];
}

def QCirc_Gate1Q1POp : QCirc_Op<"gate1q1p", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Controllable>]> {
    let summary = "A quantum gate with 1 target and 1 real-valued parameter";
    let description = [{
        Dataflow semantics one-qubit quantum gate with a variable number of
        control qubits. (That is, qubits flow through this gate like a typical
        SSA op.) There is one real-valued parameter, e.g,. the theta in
        `Rx(theta)`.
    }];
    let arguments = (ins QCirc_Gate1Q1PAttr:$gate, Builtin_Float64:$param, Variadic<QCirc_Qubit>:$controls, QCirc_Qubit:$qubit);
    let results = (outs Variadic<QCirc_Qubit>:$controlResults, QCirc_Qubit:$result);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        static llvm::StringRef getGateAttrStrName() { return "gate"; }
    }];
}

def QCirc_Gate1Q3POp : QCirc_Op<"gate1q3p", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Controllable>]> {
    let summary = "A quantum gate with 1 target and 3 real-valued parameters";
    let description = [{
        Dataflow semantics one-qubit quantum gate with a variable number of
        control qubits. (That is, qubits flow through this gate like a typical
        SSA op.) There are three real-valued parameters, e.g., in `U(theta, phi, lambda)`.
    }];
    let arguments = (ins QCirc_Gate1Q3PAttr:$gate, Builtin_Float64:$firstParam, Builtin_Float64:$secondParam, Builtin_Float64:$thirdParam, Variadic<QCirc_Qubit>:$controls, QCirc_Qubit:$qubit);
    let results = (outs Variadic<QCirc_Qubit>:$controlResults, QCirc_Qubit:$result);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        static llvm::StringRef getGateAttrStrName() { return "gate"; }
    }];
}

def QCirc_Gate2QOp : QCirc_Op<"gate2q", [
        Pure, InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<Adjointable>,
        DeclareOpInterfaceMethods<Controllable>]> {
    let summary = "A quantum gate with 2 targets and no real-valued parameters";
    let description = [{
        Dataflow semantics two-qubit quantum gate with a variable number of
        control qubits. (That is, qubits flow through this gate like a typical SSA op.)
    }];
    let arguments = (ins QCirc_Gate2QAttr:$gate, Variadic<QCirc_Qubit>:$controls, QCirc_Qubit:$leftQubit, QCirc_Qubit:$rightQubit);
    let results = (outs Variadic<QCirc_Qubit>:$controlResults, QCirc_Qubit:$leftResult, QCirc_Qubit:$rightResult);
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
        static llvm::StringRef getGateAttrStrName() { return "gate"; }
    }];
    let hasVerifier = 1;
}

def QCirc_QallocOp : QCirc_Op<"qalloc", [
        Pure,
        DeclareOpInterfaceMethods<Adjointable, ["getNumOperandsOfAdjoint"]>,
        DeclareOpInterfaceMethods<Controllable>]> {
    let summary = "Allocates and yields a fresh qubit in the zero state";
    let results = (outs Res<QCirc_Qubit, "", [MemAlloc]>:$result);
    let hasVerifier = 1;
    let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

def QCirc_MeasureOp : QCirc_Op<"measure", [Pure]> {
    let summary = "Measure a qubit in the standard basis";
    let arguments = (ins QCirc_Qubit:$qubit);
    let results = (outs QCirc_Qubit:$qubitResult, I1:$measResult);
    let hasVerifier = 1;
    let assemblyFormat = "`(` $qubit `)` attr-dict `:` functional-type(operands, results)";
}

// Intentionally not marked as Pure because we don't want the canonicalizer to
// pluck it away from us
def QCirc_QfreeOp : QCirc_Op<"qfree"> {
    let summary = "Reset and free a qubit";
    let arguments = (ins Arg<QCirc_Qubit, "", [MemFree]>:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` functional-type(operands, results)";
}

def QCirc_QfreeZeroOp : QCirc_Op<"qfreez", [
        DeclareOpInterfaceMethods<Adjointable, ["getNumOperandsOfAdjoint"]>,
        DeclareOpInterfaceMethods<Controllable>]> {
    let summary = "Free a qubit without resetting, assuming it is zero";
    let description = [{
        Means you pinky promise on the Good Book that the operand is |0⟩. In QIR
        terms, we `__quantum__rt__qubit_release()` the qubit without
        `__quantum__qis__reset__body()`ing it first
    }];
    let arguments = (ins Arg<QCirc_Qubit, "", [MemFree]>:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` functional-type(operands, results)";
}

def QCirc_QubitIndexOp : QCirc_Op<"qidx", [Pure]> {
    let summary = "Get a qubit by its statically-allocated index";
    let description = [{
        This is used only when converting to QIR with the base profile. Qallocs
        are converted to this by `BaseProfileFuncPrepPass`.
    }];
    let arguments = (ins I64Attr:$index);
    let results = (outs QCirc_Qubit:$result);
    let hasVerifier = 1;
    let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

def QCirc_UglyMeasureOp : QCirc_Op<"uglymeasure"> {
    let summary = "Measure qubits, storing the results starting at the specified result offset";
    let description = [{
        This is used only when converting to QIR with the base profile.
        `qcirc.measure`s are converted to this by `BaseProfileFuncPrepPass`.
    }];
    let arguments = (ins I64Attr:$resultOffset, Variadic<QCirc_Qubit>:$qubits);
    let assemblyFormat = "`(` $qubits `)` `offset` $resultOffset attr-dict `:` functional-type(operands, results)";
}

def QCirc_UglyRecordOp : QCirc_Op<"uglyrecord"> {
    let summary = "Record a measurement result for caller to read later";
    let description = [{
        This is used only when converting to QIR with the base profile.
        `BaseProfileFuncPrepPass` converts returning measurement results into this.
    }];
    let arguments = (ins FlatSymbolRefAttr:$uglyLabel, I64Attr:$resultOffset, I64Attr:$numResults);
    let assemblyFormat = "`named` $uglyLabel `offset` $resultOffset `for` $numResults attr-dict `:` functional-type(operands, results)";
    let builders = [
        OpBuilder<(ins "::qcirc::UglyLabelOp":$label, "uint64_t":$offset, "uint64_t":$numResults), [{
            build($_builder, $_state, mlir::FlatSymbolRefAttr::get(label), $_builder.getI64IntegerAttr(offset), $_builder.getI64IntegerAttr(numResults));
        }]>
    ];
}

def QCirc_UglyLabelOp : QCirc_Op<"uglylabel", [Symbol]> {
    let summary = "Lives at module scope. Holds the label for a recorded tuple";
    let description = [{
        This is used only when converting to QIR with the base profile.
        Will be lowered to an LLVM array constant, so must live directly inside
        the `ModuleOp`.
    }];
    let arguments = (ins StrAttr:$sym_name, StrAttr:$label);
    let assemblyFormat = "$label attr-dict";
}

def QCirc_InitOp : QCirc_Op<"init"> {
    let summary = "Initialize the quantum runtime";
    let description = [{
        This is used only when converting to QIR with the base profile.
        Lowered to `__quantum__rt__initialize()`.
    }];
    let assemblyFormat = "attr-dict";
}

def QCirc_ArrayPackOp : QCirc_Op<"arrpack", [
        Pure, InferTypeOpAdaptor, Stationary]> {
    let summary = "Allocate and initialize an array";
    let description = [{
        Lowered to a QIR array allocation routine and many store instructions.
    }];
    let arguments = (ins Variadic<AnyType>:$elems);
    let results = (outs QCirc_Array:$array);
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
    let assemblyFormat = "`(` $elems `)` attr-dict `:` functional-type(operands, results)";
}

def QCirc_ArrayUnpackOp : QCirc_Op<"arrunpack", [
        Pure, InferTypeOpAdaptor, Stationary]> {
    let summary = "Retrieve the elements of an array and free it";
    let description = [{
        Lowered to many load instructions and an QIR array refcount decrement.
    }];
    let arguments = (ins QCirc_Array:$array);
    let results = (outs Variadic<AnyType>:$elems);
    let hasCanonicalizer = 1;
    let hasVerifier = 1;
    let assemblyFormat = "$array attr-dict `:` functional-type(operands, results)";
}

def FuncSpecArrayAttr : TypedArrayAttrBase<
    QCirc_FuncSpecAttr,
    "Array of function specializations">;

def QCirc_CallableMetadataOp : QCirc_Op<"callable_metadata", [Symbol]> {
    let summary = "Lives at module scope. Holds QIR callable metadata";
    let description = [{
        Lowered to a QIR callable implementation table and memory management
        table.
    }];
    let arguments = (ins StrAttr:$sym_name,
                         TypeArrayAttr:$capture_types,
                         FuncSpecArrayAttr:$specs);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        auto getCaptureTypeRange() {
            return getCaptureTypes().getAsValueRange<mlir::TypeAttr>();
        }
        auto getSpecsRange() {
            return getSpecs().getAsRange<FuncSpecAttr>();
        }
    }];
}

def QCirc_CallableCreateOp : QCirc_Op<"callable_create", [
        Pure,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Create a QIR callable value";
    let description = [{
        Lowered to `__quantum__rt__callable_create()`
    }];
    let arguments = (ins FlatSymbolRefAttr:$metadata, Variadic<AnyType>:$captures);
    let assemblyFormat = "$metadata `[` $captures `]` attr-dict `:` functional-type(operands, results)";
    let results = (outs QCirc_Callable:$callable);
}

def QCirc_CallableAdjointOp : QCirc_Op<"callable_adj", [
        Pure, SameOperandsAndResultType]> {
    let summary = "Yield an adjoint version of this callable value";
    let description = [{
        Lowers to copying the callable and then calling `__quantum__rt__callable_make_adjoint()`
    }];
    let arguments = (ins QCirc_Callable:$callableIn);
    let assemblyFormat = "$callableIn attr-dict `:` functional-type(operands, results)";
    let results = (outs QCirc_Callable:$callableOut);
}

def QCirc_CallableControlOp : QCirc_Op<"callable_ctrl", [Pure]> {
    let summary = "Yield a controlled version of this callable value";
    let description = [{
        Lowers to copying the callable and then calling `__quantum__rt__callable_make_controlled()`
    }];
    let arguments = (ins QCirc_Callable:$callableIn);
    let assemblyFormat = "$callableIn attr-dict `:` functional-type(operands, results)";
    let results = (outs QCirc_Callable:$callableOut);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$callableIn, "uint64_t":$numControls), [{
            CallableType operand_ty =
                llvm::cast<CallableType>(callableIn.getType());
            mlir::FunctionType inner_func_ty = operand_ty.getFunctionType();
            assert(inner_func_ty.getNumInputs() == 1
                   && inner_func_ty.getInputs() == inner_func_ty.getResults()
                   && "Invariants of reversible function violated");
            ArrayType arg_ty =
                llvm::dyn_cast<ArrayType>(inner_func_ty.getInputs()[0]);
            assert(arg_ty
                   && llvm::isa<QubitType>(arg_ty.getElemType())
                   && "Reversible function must take Array<Qubit>");
            ArrayType new_arg_ty =
                $_builder.getType<ArrayType>(
                    arg_ty.getElemType(), arg_ty.getDim() + numControls);
            CallableType result_ty =
                $_builder.getType<CallableType>(
                    $_builder.getType<mlir::FunctionType>(
                        new_arg_ty,
                        new_arg_ty));

            build($_builder, $_state, result_ty, callableIn);
        }]>
    ];
}

def QCirc_CallableInvokeOp : QCirc_Op<"callable_invoke", [
        InferTypeOpAdaptor]> {
    let summary = "Call a callable value";
    let description = [{
        Lowers to calling `__quantum__rt__callable_invoke()`
    }];
    let arguments = (ins QCirc_Callable:$callable,
                         Variadic<AnyType>:$call_operands);
    let results = (outs Variadic<AnyType>:$results);
    let assemblyFormat = "$callable `(` $call_operands `)` attr-dict `:` functional-type(operands, results)";
}

def QCirc_CallableCopyOp : QCirc_Op<"callable_copy", [
        Pure, SameOperandsAndResultType]> {
    let summary = "Copy a callable value";
    let description = [{
        Inserted automatically by `QCircToQIRConversionPass`
    }];
    let arguments = (ins QCirc_Callable:$callableIn);
    let assemblyFormat = "$callableIn attr-dict `:` functional-type(operands, results)";
    let results = (outs QCirc_Callable:$callableOut);
}

def QCirc_CallableFreeOp : QCirc_Op<"callable_free"> {
    let summary = "Free a callable value";
    let description = [{
        Inserted automatically by `QCircToQIRConversionPass`
    }];
    let arguments = (ins QCirc_Callable:$callable);
    let assemblyFormat = "$callable attr-dict `:` qualified(type($callable))";
}

def QCirc_ArrayCopyOp : QCirc_Op<"array_copy", [
        Pure, SameOperandsAndResultType]> {
    let summary = "Copy an array";
    let description = [{
        Inserted automatically by `QCircToQIRConversionPass`
    }];
    let arguments = (ins QCirc_Array:$arrayIn);
    let assemblyFormat = "$arrayIn attr-dict `:` functional-type(operands, results)";
    let results = (outs QCirc_Array:$arrayOut);
}

def QCirc_ArrayFreeOp : QCirc_Op<"array_free"> {
    let summary = "Free an array";
    let description = [{
        Inserted automatically by `QCircToQIRConversionPass`
    }];
    let arguments = (ins QCirc_Array:$array);
    let assemblyFormat = "$array attr-dict `:` qualified(type($array))";
}

def QCirc_LLVMConstantArrayOp : QCirc_Op<"llvm_array", [Pure]> {
    let summary = "Constant LLVM Array";
    let description = [{
        This is a truly awful op that does not belong in this dialect. It is a
        hack to generate an `llvm::ConstantArray` containing `!llvm.ptr`s, which
        surprisingly the `llvm` dialect does not support on its own today (LLVM
        19.1.2). We should try to merge this into the upstream `llvm' dialect
        ASAP, but for now, it is easier to avoid forking LLVM. Once the fix is
        merged upstream, we can remove this op, the `qwerty-translate` tool,
        `QCircLLVMIRTranslationInterface`, and `registerQCircDialectTranslation`.
    }];
    let arguments = (ins Variadic<AnyType>:$elements);
    let results = (outs AnyType:$result);
    let assemblyFormat = "`(` $elements `)` attr-dict `:` functional-type(operands, results)";
}

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_OPS_TD
