#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "Qwerty/IR/QwertyOps.h"
#include "QCirc/IR/QCircOps.h"

#include "defs.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"
#include "mlir_handle.hpp"

bool QpuLoweringVisitor::visitNonQpuNode(ASTVisitContext &ctx, ASTNode &node) {
    throw CompileException("How did a non-" + ast_kind_name(AST_QPU) + " node "
                           "end up in a " + ast_kind_name(AST_QPU) + " AST?",
                           std::move(node.dbg->copy()));
}

mlir::ValueRange QpuLoweringVisitor::wrapStationary(mlir::Location loc,
                                                    mlir::TypeRange result_types,
                                                    mlir::ValueRange args,
                                                    std::function<void(mlir::ValueRange)> build_body) {
    qcirc::CalcOp calc = handle.builder.create<qcirc::CalcOp>(loc, result_types, args);
    {
        mlir::OpBuilder::InsertionGuard guard(handle.builder);
        // Sets insertion point to end of this block
        llvm::SmallVector<mlir::Location> arg_locs(args.size(), loc);
        mlir::Block *calc_block = handle.builder.createBlock(&calc.getRegion(), {}, args.getTypes(), arg_locs);
        assert(calc_block->getNumArguments() == args.size());
        build_body(calc_block->getArguments());
    }
    mlir::ValueRange calc_results = calc.getResults();
    assert(calc_results.size() == result_types.size());
    return calc_results;
}

mlir::Value QpuLoweringVisitor::materializeSimpleCapture(DebugInfo &dbg, mlir::Location loc, HybridObj *capture) {
    if (Bits *bits = dynamic_cast<Bits *>(capture)) {
        [[maybe_unused]] const BitType &bitType =
            dynamic_cast<const BitType &>(bits->getType());
        assert(bitType.dim->isConstant()
               && "type variables remaining in compilation!");
        assert(bitType.dim->isConstant()
               && bitType.dim->offset == (DimVarValue)bits->bits.size()
               && "Type of Bits does not match its actual number of bits");

        size_t n_bits = bits->bits.size();
        mlir::Type stat_type = handle.builder.getType<qwerty::BitBundleType>(n_bits);
        mlir::ValueRange stat_vals = wrapStationary(loc, stat_type, {}, [&](mlir::ValueRange args) {
            assert(args.empty());
            llvm::SmallVector<mlir::Value> bits_to_bundle(n_bits);
            for (size_t j = 0; j < bits->bits.size(); j++) {
                mlir::Value bit = handle.builder.create<mlir::arith::ConstantOp>(loc, handle.builder.getBoolAttr(bits->bits[j])).getResult();
                bits_to_bundle[j] = bit;
            }
            mlir::Value bits_bundled =
                handle.builder.create<qwerty::BitBundlePackOp>(loc, bits_to_bundle).getBundle();
            handle.builder.create<qcirc::CalcYieldOp>(loc, bits_bundled);
        });
        assert(stat_vals.size() == 1);
        return stat_vals[0];
    } else if (Integer *integer = dynamic_cast<Integer *>(capture)) {
        mlir::ValueRange stat_vals = wrapStationary(loc, handle.builder.getI64Type(), {}, [&](mlir::ValueRange args) {
            assert(args.empty());
            mlir::Value val = handle.builder.create<mlir::arith::ConstantOp>(loc, handle.builder.getI64IntegerAttr(integer->val));
            handle.builder.create<qcirc::CalcYieldOp>(loc, val);
        });
        assert(stat_vals.size() == 1);
        return stat_vals[0];
    } else if (Angle *angle = dynamic_cast<Angle *>(capture)) {
        mlir::ValueRange stat_vals = wrapStationary(loc, handle.builder.getF64Type(), {}, [&](mlir::ValueRange args) {
            assert(args.empty());
            mlir::Value val = handle.builder.create<mlir::arith::ConstantOp>(loc, handle.builder.getF64FloatAttr(angle->val));
            handle.builder.create<qcirc::CalcYieldOp>(loc, val);
        });
        assert(stat_vals.size() == 1);
        return stat_vals[0];
    } else {
        // TODO: This either needs to be caught by the type system, or we
        //       need to allow you to do crazier stuff
        throw CompileException("Currently, " + ast_kind_name(AST_QPU)
                               + " kernels can only capture bit[N], int, "
                               " float/angle, and other kernels. Sorry!",
                               std::move(dbg.copy()));
    }
}

mlir::Value QpuLoweringVisitor::createLambda(mlir::Location loc,
                                             const FuncType &func_type,
                                             mlir::ValueRange captures,
                                             std::function<void(mlir::ValueRange, mlir::ValueRange)> add_contents) {
    llvm::SmallVector<mlir::Type> func_type_types = func_type.toMlirType(handle);
    assert(func_type_types.size() == 1
           && llvm::isa<qwerty::FunctionType>(func_type_types[0])
           && "FuncType turned into something other than a qwerty::FunctionType");

    qwerty::FunctionType lambda_func_type = llvm::cast<qwerty::FunctionType>(func_type_types[0]);
    qwerty::LambdaOp lambda = handle.builder.create<qwerty::LambdaOp>(loc, lambda_func_type, captures);

    // ===> Save insertion point
    auto old_insertpt = handle.builder.saveInsertionPoint();

    // First N block arguments are captures
    size_t n_captures = captures.size();
    llvm::SmallVector<mlir::Location> block_arg_locs(
        n_captures + lambda_func_type.getFunctionType().getInputs().size(),
        loc);
    llvm::SmallVector<mlir::Type> block_arg_types(
        captures.getTypes().begin(), captures.getTypes().end());
    block_arg_types.append(
        lambda_func_type.getFunctionType().getInputs().begin(),
        lambda_func_type.getFunctionType().getInputs().end());
    // Sets insert point to end of this block
    mlir::Block *entry_block = handle.builder.createBlock(
            &lambda.getRegion(), {},
            block_arg_types, block_arg_locs);

    llvm::SmallVector<mlir::Value> capture_block_args(
        entry_block->args_begin(),
        entry_block->args_begin() + n_captures);
    llvm::SmallVector<mlir::Value> func_arg_block_args(
        entry_block->args_begin() + n_captures,
        entry_block->args_end());
    add_contents(capture_block_args, func_arg_block_args);

    // <=== Restore old insertion point
    handle.builder.restoreInsertionPoint(old_insertpt);

    return lambda;
}

void QpuLoweringVisitor::init(ASTNode &root) {
    QpuKernel &kernel = dynamic_cast<QpuKernel &>(root);
    temp_func_prefix = "__" + funcOp_name + "_";
    n_temp_funcs = 0;

    assert(provided_captures.size() == kernel.capture_names.size()
           && provided_captures.size() == kernel.capture_types.size()
           && "capture length mismatch");

    mlir::Location loc = kernel.dbg->toMlirLoc(handle);
    // TODO: catch when a function returns a qubit or a basis, since we can
    //       handle neither properly right now
    llvm::SmallVector<mlir::Type> kernel_types = kernel.getType().toMlirType(handle);
    assert(kernel_types.size() == 1
           && llvm::isa<qwerty::FunctionType>(kernel_types[0])
           && "Kernel MLIR type should be 1 FunctionType");
    qwerty::FunctionType func_type = llvm::cast<qwerty::FunctionType>(kernel_types[0]);
    mlir::FunctionType inner_func_type = func_type.getFunctionType();

    handle.builder.setInsertionPointToEnd(handle.module->getBody());
    qwerty::FuncOp func = handle.builder.create<qwerty::FuncOp>(loc, funcOp_name, func_type);
    llvm::SmallVector<mlir::Location> arg_locs(inner_func_type.getNumInputs(), loc);
    // Sets insert point to end of this block
    mlir::Block *entryBlock = handle.builder.createBlock(
        &func.getBody(), {}, inner_func_type.getInputs(), arg_locs);
    kernel.funcOp = func;

    // Set up symbol table for walking body
    for (size_t i = 0; i < kernel.arg_names.size(); i++) {
        [[maybe_unused]] bool added = variable_values.emplace(
                kernel.arg_names[i],
                std::initializer_list<mlir::Value>{entryBlock->getArgument(i)}
            ).second;
        assert(added && "Duplicate argument names. How did this pass typechecking?");
    }

    // Now set up captures
    assert(provided_captures.size() == kernel.capture_names.size()
           && provided_captures.size() == kernel.capture_types.size()
           && "capture length mismatch");

    // Synthesize constant captures and reference FuncOps for other kernels
    for (size_t i = 0; i < provided_captures.size(); i++) {
        if (ClassicalKernel *classical_kernel = dynamic_cast<ClassicalKernel *>(provided_captures[i].get())) {
            if (!classical_kernel->embedding_func_ops.empty()) {
                [[maybe_unused]] bool added = cfunc_names.emplace(
                        std::make_pair(kernel.capture_names[i], std::vector<DimVarValue>{}),
                        classical_kernel
                    ).second;
                assert(added && "Duplicate name of classical kernel. "
                                "How did AST typechecking allow this?");
            } else {
                // This is probably instantiated. Let the EmbedClassical visitor
                // know by setting the pointer to null
                [[maybe_unused]] bool added = cfunc_names.emplace(
                    std::make_pair(kernel.capture_names[i], std::vector<DimVarValue>{}),
                    nullptr
                ).second;
                assert(added && "Classical function name duplicated. How is this possible?");

                for (auto &kv : *kernel.capture_instances[i]) {
                    const std::vector<DimVarValue> &instance_vals = kv.first;
                    ClassicalKernel *instance_kernel = dynamic_cast<ClassicalKernel *>(kv.second.get());
                    assert(instance_kernel && "Instance of classical function is not a classical function, huh?");
                    assert(!instance_kernel->embedding_func_ops.empty() && "Instantiated classical kernel missing funcOps! How?");
                    [[maybe_unused]] bool added = cfunc_names.emplace(
                        std::make_pair(kernel.capture_names[i], instance_vals),
                        instance_kernel
                    ).second;
                    assert(added && "Duplicate instantiated capture. How is this possible?");
                }
            }
        } else if (Kernel *callee = dynamic_cast<Kernel *>(provided_captures[i].get())) {
            if (callee->funcOp) {
                assert(callee->funcOp && "No FuncOp for callee yet. How?");
                [[maybe_unused]] bool added = variable_values.emplace(
                        kernel.capture_names[i],
                        std::initializer_list<mlir::Value>{
                            handle.builder.create<qwerty::FuncConstOp>(loc, callee->funcOp).getResult()}
                    ).second;
                assert(added && "Duplicate argument name between captures and "
                                "arguments. How did AST typechecking allow this?");
            } else if (!variable_values.count(kernel.capture_names[i])) {
                // Make the variable visitor shut up in the case of Instantiates
                variable_values.emplace(kernel.capture_names[i], std::initializer_list<mlir::Value>{});
            }
            // TODO: don't just access random fields of Kernel here, pass this
            // in somehow instead
            for (auto &kv : *kernel.capture_instances[i]) {
                const std::vector<DimVarValue> &instance_vals = kv.first;
                Kernel &instance_kernel = dynamic_cast<Kernel &>(*kv.second);
                assert(instance_kernel.funcOp && "Instantiated kernel missing funcOp! How?");
                [[maybe_unused]] bool added = instantiate_values.emplace(
                        std::make_pair(kernel.capture_names[i], instance_vals),
                        std::initializer_list<mlir::Value>{
                           handle.builder.create<qwerty::FuncConstOp>(loc, instance_kernel.funcOp).getResult()}
                    ).second;
                assert(added && "Duplicate instantiated capture. How is this possible?");
            }
        } else if (Tuple *tuple = dynamic_cast<Tuple *>(provided_captures[i].get())) {
            if (std::unique_ptr<BroadcastType> complex_array =
                    tuple->getType().collapseToHomogeneousArray<ComplexType>()) {
                std::vector<std::complex<double>> state_vec;
                assert(complex_array->factor->isConstant()
                       && "dimvars snuck into statevector capture lowering");
                state_vec.reserve(complex_array->factor->offset);
                assert((size_t)complex_array->factor->offset == tuple->children.size()
                       && "mismatch in type and number of children for "
                          "complex tuple");

                for (std::unique_ptr<HybridObj> &child : tuple->children) {
                    Amplitude *amp = dynamic_cast<Amplitude *>(child.get());
                    assert(amp && "Statevector contains something other than "
                                  "an amplitude, how?");
                    state_vec.push_back(amp->val);
                }

                [[maybe_unused]] bool added = amplitude_names.emplace(
                        kernel.capture_names[i], std::move(state_vec)
                    ).second;
                assert(added && "Duplicate name of statevector");

                // Make the variable visitor shut up
                variable_values.emplace(kernel.capture_names[i],
                                        std::initializer_list<mlir::Value>{});
            } else {
                // TODO: move the {Classical,}Kernel stuff above into
                //       materializeSimpleCapture. That way you can capture tuples
                //       of kernels
                llvm::SmallVector<mlir::Value> values;
                values.reserve(tuple->children.size());
                for (size_t j = 0; j < tuple->children.size(); j++) {
                    values.push_back(materializeSimpleCapture(*root.dbg, loc, tuple->children[j].get()));
                }
                [[maybe_unused]] bool added = variable_values.emplace(
                        kernel.capture_names[i],
                        values
                    ).second;
                assert(added && "Duplicate capture/argument name. How?");
            }
        } else {
            mlir::Value val = materializeSimpleCapture(*root.dbg, loc, provided_captures[i].get());
            [[maybe_unused]] bool added = variable_values.emplace(
                    kernel.capture_names[i],
                    std::initializer_list<mlir::Value>{val}
                ).second;
            assert(added && "Duplicate capture/argument name. How?");
        }
    }
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Assign &assign) {
    assign.value->visit(ctx, *this);
    variable_values[assign.target] = assign.value->getMlirValues();
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, DestructAssign &dassign) {
    dassign.value->visit(ctx, *this);

    mlir::Location loc = dassign.dbg->toMlirLoc(handle);
    const Type &rhs_ty = dassign.value->getType();
    ASTNode::Values &mlir_values = dassign.value->getMlirValues();

    if (dynamic_cast<const TupleType *>(&rhs_ty)) {
        assert(mlir_values.size() == dassign.targets.size()
               && "Mismatch in dimensions of lhs and rhs of destructuring "
                  "assignment");
        for (size_t i = 0; i < dassign.targets.size(); i++) {
            variable_values[dassign.targets[i]] = {mlir_values[i]};
        }
    } else if (dynamic_cast<const QubitType *>(&rhs_ty)) {
        assert(mlir_values.size() == 1);
        mlir::ValueRange qubits =
            handle.builder.create<qwerty::QBundleUnpackOp>(
                loc, mlir_values[0]).getQubits();
        assert(qubits.size() == dassign.targets.size());
        for (size_t i = 0; i < qubits.size(); i++) {
            variable_values[dassign.targets[i]] =
                {handle.builder.create<qwerty::QBundlePackOp>(
                    loc, qubits[i]).getQbundle()};
        }
    } else if (dynamic_cast<const BitType *>(&rhs_ty)) {
        assert(mlir_values.size() == 1);
        mlir::ValueRange bits =
            handle.builder.create<qwerty::BitBundleUnpackOp>(
                loc, mlir_values[0]).getBits();
        assert(bits.size() == dassign.targets.size());
        for (size_t i = 0; i < bits.size(); i++) {
            variable_values[dassign.targets[i]] =
                {handle.builder.create<qwerty::BitBundlePackOp>(
                    loc, bits[i]).getBundle()};
        }
    } else {
        assert(0 && "Unknown rhs of destructuring assignment");
    }

    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Return &ret) {
    ret.value->visit(ctx, *this);
    mlir::Location loc = ret.dbg->toMlirLoc(handle);
    handle.builder.create<qwerty::ReturnOp>(loc, ret.value->getMlirValues());
    return true;
}

// Everything to do here was already handled by init()
bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Kernel &kernel) {
    for (size_t i = 0; i < kernel.body.size(); i++) {
        ASTNode *node = kernel.body[i].get();
        node->visit(ctx, *this);
    }
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Adjoint &adj) {
    adj.operand->visit(ctx, *this);
    mlir::Location loc = adj.dbg->toMlirLoc(handle);

    assert(adj.operand->getMlirValues().size() == 1 && "Operand of adjoint should have 1 MLIR Value");
    const FuncType *func_type = dynamic_cast<const FuncType *>(&adj.operand->getType());
    assert(func_type && "operand of adjoint should have func type");
    [[maybe_unused]] const QubitType *qubit_lhs =
        dynamic_cast<const QubitType *>(func_type->lhs.get());
    [[maybe_unused]] const QubitType *qubit_rhs =
        dynamic_cast<const QubitType *>(func_type->rhs.get());
    assert(qubit_lhs && qubit_rhs
           && "operand of adjoint should take qubits as input");
    assert(qubit_lhs->dim->isConstant()
           && qubit_rhs->dim->isConstant()
           && "dimvars snuck into adjoint lowering");
    assert(qubit_lhs->dim->offset == qubit_rhs->dim->offset
           && "mismatch in number of qubits in adjoint operand");

    mlir::Value func = adj.operand->getMlirValues()[0];
    mlir::Value adjointed = handle.builder.create<qwerty::FuncAdjointOp>(
        loc, func).getResult();
    adj.getMlirValues() = {adjointed};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Prepare &prep) {
    prep.operand->visit(ctx, *this);

    mlir::Location loc = prep.dbg->toMlirLoc(handle);

    const FuncType *func_type = dynamic_cast<const FuncType *>(&prep.getType());
    assert(func_type && "prep is not a function??");

    mlir::Value lambda;
    if (dynamic_cast<const BitType *>(&prep.operand->getType())) {
        assert(prep.operand->getMlirValues().size() == 1
               && "Operand of Prepare should have 1 MLIR Value");
        mlir::Value operand = prep.operand->getMlirValues()[0];
        auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
            assert(captures.size() == 1 && args.size() == 1);
            mlir::Value bits = captures[0];
            mlir::Value qbundle_in = args[0];
            mlir::Value ret = handle.builder.create<qwerty::BitInitOp>(loc, bits, qbundle_in).getQbundleOut();
            handle.builder.create<qwerty::ReturnOp>(loc, ret);
        };
        lambda = createLambda(loc, *func_type, operand, lambda_contents);
    } else if (dynamic_cast<const BasisType *>(&prep.operand->getType())) {
        ASTNode::BasisValue &basis = prep.operand->getBasis();
        auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
            assert(args.size() == 1);
            mlir::Value qbundle_in = args[0];
            mlir::Value ret = handle.builder.create<qwerty::QBundleInitOp>(loc, basis.basis, captures, qbundle_in).getQbundleOut();
            handle.builder.create<qwerty::ReturnOp>(loc, ret);
        };
        lambda = createLambda(loc, *func_type, basis.phases, lambda_contents);
    } else {
        assert(0 && "operand of Prepare is neither a basis nor a bit[N]");
    }

    prep.getMlirValues() = {lambda};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Lift &lift) {
    assert(lift.operand->getType().collapseToHomogeneousArray<ComplexType>()
           && "Lift should be taken care of canonicalizer, how is it "
              "reaching qpu_lowering?");
    Variable *var = dynamic_cast<Variable *>(lift.operand.get());
    assert(var && "Operand of lift should be a Variable");
    assert(amplitude_names.count(var->name)
           && "Missing state vector for lift");

    std::vector<std::complex<double>> &state_vec =
        amplitude_names.at(var->name);

    const QubitType *qubit_ty =
        dynamic_cast<const QubitType *>(&lift.getType());
    assert(qubit_ty && "Result of lift is not Qubit[N]");
    assert(qubit_ty->dim->isConstant() && "Dimvars snuck into Lift type");
    size_t dim = qubit_ty->dim->offset;

    llvm::SmallVector<qwerty::SuperposElemAttr> elems;
    for (size_t i = 0; i < state_vec.size(); i++) {
        double prob = std::pow(std::abs(state_vec[i]), 2.0);
        double tilt = std::arg(state_vec[i]);
        llvm::APInt eigenbits(/*numBits=*/dim, /*val=*/i, /*isSigned=*/false);

        elems.push_back(handle.builder.getAttr<qwerty::SuperposElemAttr>(
            handle.builder.getF64FloatAttr(prob),
            handle.builder.getF64FloatAttr(tilt),
            handle.builder.getAttr<qwerty::BasisVectorAttr>(
                qwerty::PrimitiveBasis::Z,
                eigenbits,
                dim,
                /*hasPhase=*/false)));
    }

    qwerty::SuperposAttr superpos =
        handle.builder.getAttr<qwerty::SuperposAttr>(elems);
    mlir::Location loc = lift.dbg->toMlirLoc(handle);
    lift.getMlirValues() = std::initializer_list<mlir::Value>{
        handle.builder.create<qwerty::SuperposOp>(loc, superpos).getQbundle()};

    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, EmbedClassical &embed) {
    assert(cfunc_names.count(std::make_pair(embed.name, std::vector<DimVarValue>{}))
           && "Classical function variable missing from symbol table. "
              "How did this pass AST typechecking?");

    if (!cfunc_names.at(std::make_pair(embed.name, std::vector<DimVarValue>{}))) {
        // This is inside a instantation. Shut up!
        embed.getMlirValues() = std::initializer_list<mlir::Value>{};
        return true;
    } else {
        mlir::Location loc = embed.dbg->toMlirLoc(handle);
        ClassicalKernel *operand = nullptr;
        if (!embed.operand_name.empty()) {
            assert(cfunc_names.count(std::make_pair(embed.operand_name, std::vector<DimVarValue>{}))
                   && "Reversible function missing from symbol table.");
            operand = cfunc_names.at(std::make_pair(embed.operand_name, std::vector<DimVarValue>{}));
        }

        qwerty::FuncOp embedded_func =
            cfunc_names.at(std::make_pair(embed.name, std::vector<DimVarValue>{}))->getFuncOp(handle, operand, embed.kind);
        embed.getMlirValues() = std::initializer_list<mlir::Value>{
            handle.builder.create<qwerty::FuncConstOp>(loc, embedded_func).getResult()};
        return true;
    }
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Variable &var) {
    assert(variable_values.count(var.name) && "Variable missing from symbol table. "
                                              "How did this pass AST typechecking?");
    var.getMlirValues() = variable_values[var.name];
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Pipe &pipe) {
    pipe.left->visit(ctx, *this);
    pipe.right->visit(ctx, *this);
    mlir::Location loc = pipe.dbg->toMlirLoc(handle);
    assert(pipe.right->getMlirValues().size() == 1
           && llvm::isa<qwerty::FunctionType>(pipe.right->getMlirValues()[0].getType())
           && "Right-hand side of pipe is not a function");
    mlir::Value func = pipe.right->getMlirValues()[0];
    pipe.getMlirValues() = handle.builder.create<qwerty::CallIndirectOp>(loc, func, pipe.left->getMlirValues()).getResults();
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Instantiate &inst) {
    inst.var->visit(ctx, *this);
    std::vector<DimVarValue> instance_vals;
    instance_vals.reserve(inst.instance_vals.size());

    for (std::unique_ptr<DimVarExpr> &val : inst.instance_vals) {
        assert(val->isConstant() && "dimvars snuck into instantiate val");
        instance_vals.push_back(val->offset);
    }

    std::string name;
    if (Variable *var = dynamic_cast<Variable *>(inst.var.get())) {
        name = var->name;
        auto pair = std::make_pair(name, instance_vals);
        assert(instantiate_values.count(pair) && "Instantiate value not map, what's going on?");
        inst.getMlirValues() = instantiate_values[pair];
    } else if (EmbedClassical *embed = dynamic_cast<EmbedClassical *>(inst.var.get())) {
        name = embed->name;
        assert(cfunc_names.count(std::make_pair(name, instance_vals))
               && "Instantiate value not map, what's going on?");
        mlir::Location loc = inst.dbg->toMlirLoc(handle);

        ClassicalKernel *operand = nullptr;
        if (!embed->operand_name.empty()) {
            assert(cfunc_names.count(std::make_pair(embed->operand_name,
                                                    instance_vals))
                   && "Operand instantiation value not map, what's going on?");
            operand = cfunc_names.at(std::make_pair(embed->operand_name, instance_vals));
        }

        qwerty::FuncOp embedded_func =
            cfunc_names.at(std::make_pair(name, instance_vals))->getFuncOp(handle, operand, embed->kind);
        inst.getMlirValues() = std::initializer_list<mlir::Value>{
            handle.builder.create<qwerty::FuncConstOp>(loc, embedded_func).getResult()};
    } else {
        assert(0 && "Instantiate var is neither a variable nor embedded classical "
                    "func, how did this pass typechecking?");
    }
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Repeat &repeat) {
    // Repeat constructs should be unrolled by EvalDimVarExprVisitor
    throw CompileException("Repeat construct made it to lowering. This is a "
                           "compiler bug.",
                           std::move(repeat.dbg->copy()));
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, RepeatTensor &reptens) {
    // RepeatTensor constructs should be unrolled by EvalDimVarExprVisitor
    throw CompileException("RepeatTensor construct made it to lowering. This "
                           "is a compiler bug.",
                           std::move(reptens.dbg->copy()));
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Pred &pred) {
    pred.basis->visit(ctx, *this);
    pred.body->visit(ctx, *this);
    mlir::Location loc = pred.dbg->toMlirLoc(handle);
    assert(pred.body->getMlirValues().size() == 1
           && "Wrong number of Values for body of Pred");
    mlir::Value func = pred.body->getMlirValues()[0];
    ASTNode::BasisValue &basis = pred.basis->getBasis();

    mlir::Value func_pred =
        handle.builder.create<qwerty::FuncPredOp>(
            loc, basis.basis.deletePhases(), func).getResult();

    // If someone wrote f & b (instead of b & f), we need to slip in a lambda
    // to shuffle the qubits around
    if (pred.order == PRED_ORDER_U_B) {
        const Type &type = pred.getType();
        const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
        size_t pred_dim = basis.basis.getDim();

        auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
            assert(captures.size() == 1 && args.size() == 1);
            mlir::Value capture_func = captures[0];
            mlir::Value arg = args[0];

            mlir::ValueRange arg_unpacked =
                handle.builder.create<qwerty::QBundleUnpackOp>(
                    loc, arg).getQubits();
            size_t meat_dim = arg_unpacked.size()-pred_dim;
            llvm::SmallVector<mlir::Value> shuffled(
                arg_unpacked.begin()+meat_dim,
                arg_unpacked.end());
            shuffled.append(arg_unpacked.begin(),
                            arg_unpacked.begin()+meat_dim);
            mlir::Value repacked =
                handle.builder.create<qwerty::QBundlePackOp>(
                    loc, shuffled).getQbundle();
            mlir::ValueRange outputs =
                handle.builder.create<qwerty::CallIndirectOp>(
                    loc, capture_func, repacked).getResults();
            assert(outputs.size() == 1);
            mlir::Value output = outputs[0];
            mlir::ValueRange output_unpacked =
                handle.builder.create<qwerty::QBundleUnpackOp>(
                    loc, output).getQubits();

            llvm::SmallVector<mlir::Value> unshuffled(
                output_unpacked.begin() + pred_dim,
                output_unpacked.end());
            unshuffled.append(output_unpacked.begin(),
                              output_unpacked.begin() + pred_dim);
            mlir::Value final_repacked =
                handle.builder.create<qwerty::QBundlePackOp>(
                    loc, unshuffled).getQbundle();
            handle.builder.create<qwerty::ReturnOp>(loc, final_repacked);
        };

        func_pred = createLambda(loc, *funcType, func_pred, lambda_contents);
    }

    pred.getMlirValues() = {func_pred};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, BiTensor &bitensor) {
    bitensor.left->visit(ctx, *this);
    bitensor.right->visit(ctx, *this);
    mlir::Location loc = bitensor.dbg->toMlirLoc(handle);
    const Type &type = bitensor.getType();
    const FuncType *funcType;

    enum class RegType {BIT, QUBIT, UNIT};
    auto get_dim = [&](const Type &type) {
        uint64_t ret = 0;
        RegType reg_type = RegType::UNIT;
        const TupleType *tupleType;
        if (const QubitType *qubitType = dynamic_cast<const QubitType *>(&type)) {
            assert(qubitType->dim->isConstant() && "dimvars snuck into mlir lowering?");
            ret = qubitType->dim->offset;
            reg_type = RegType::QUBIT;
        } else if (const BitType *bitType = dynamic_cast<const BitType *>(&type)) {
            assert(bitType->dim->isConstant() && "dimvars snuck into mlir lowering?");
            ret = bitType->dim->offset;
            reg_type = RegType::BIT;
        } else if (!(tupleType = dynamic_cast<const TupleType *>(&type))
                || !tupleType->isUnit()) {
            throw CompileException("Sorry, I don't know how to get the "
                                   "dimension of " + type.toString(),
                                   std::move(bitensor.dbg->copy()));
        }
        return std::make_pair(ret, reg_type);
    };
    auto compat_regs = [&](RegType left, RegType right) {
        return (left == RegType::UNIT || right == RegType::UNIT)
               || left == right;
    };

    if (dynamic_cast<const BitType *>(&type)) {
        // TODO: do this the same way as qbundles
        throw CompileException("Sorry, I don't know how to BiTensor bit[N]s yet",
                               std::move(bitensor.dbg->copy()));
    } else if (dynamic_cast<const BasisType *>(&type)) {
        ASTNode::BasisValue &left_basis = bitensor.left->getBasis();
        ASTNode::BasisValue &right_basis = bitensor.right->getBasis();
        ASTNode::BasisValue &bitensor_basis = bitensor.getBasis();

        if (!left_basis.basis && !right_basis.basis) {
            bitensor_basis.basis = nullptr;
            bitensor_basis.phases.clear();
        } else if (!left_basis.basis) {
            bitensor_basis = right_basis;
        } else if (!right_basis.basis) {
            bitensor_basis = left_basis;
        } else {
            llvm::SmallVector<qwerty::BasisElemAttr> new_elems(
                left_basis.basis.getElems().begin(),
                left_basis.basis.getElems().end());
            new_elems.append(
                right_basis.basis.getElems().begin(),
                right_basis.basis.getElems().end());
            bitensor_basis.basis =
                handle.builder.getAttr<qwerty::BasisAttr>(new_elems);
            bitensor_basis.phases.clear();
            bitensor_basis.phases.append(left_basis.phases.begin(),
                                         left_basis.phases.end());
            bitensor_basis.phases.append(right_basis.phases.begin(),
                                         right_basis.phases.end());
        }
    } else if (dynamic_cast<const QubitType *>(&type)) {
        llvm::SmallVector<mlir::Value> &left = bitensor.left->getMlirValues();
        llvm::SmallVector<mlir::Value> &right = bitensor.right->getMlirValues();

        if (!left.size() && !right.size()) {
            bitensor.getMlirValues() = {};
        } else if (!left.size()) {
            bitensor.getMlirValues() = right;
        } else if (!right.size()) {
            bitensor.getMlirValues() = left;
        } else {
            // Can't cheese it, time to bite the bullet and unpack/repack
            assert(left.size() == 1
                   && right.size() == 1
                   && "Expected a single MLIR Value for both sides of BiTensor");
            mlir::ValueRange left_unpack = handle.builder.create<qwerty::QBundleUnpackOp>(loc, left[0]).getQubits();
            mlir::ValueRange right_unpack = handle.builder.create<qwerty::QBundleUnpackOp>(loc, right[0]).getQubits();
            llvm::SmallVector<mlir::Value> merged;
            merged.append(left_unpack.begin(), left_unpack.end());
            merged.append(right_unpack.begin(), right_unpack.end());
            bitensor.getMlirValues() = {handle.builder.create<qwerty::QBundlePackOp>(loc, merged).getQbundle()};
        }
    } else if ((funcType = dynamic_cast<const FuncType *>(&type))) {
               //&& dynamic_cast<const QubitType *>(funcType->lhs.get())
               //&& dynamic_cast<const QubitType *>(funcType->rhs.get())) {
        auto [dim_in, reg_type_in] = get_dim(*funcType->lhs);
        auto [dim_out, reg_type_out] = get_dim(*funcType->rhs);
        const FuncType &leftFuncType = dynamic_cast<const FuncType &>(bitensor.left->getType());
        auto [left_dim_in, left_reg_type_in] = get_dim(*leftFuncType.lhs);
        auto [left_dim_out, left_reg_type_out] = get_dim(*leftFuncType.rhs);
        const FuncType &rightFuncType = dynamic_cast<const FuncType &>(bitensor.right->getType());
        auto [right_dim_in, right_reg_type_in] = get_dim(*rightFuncType.lhs);
        auto [right_dim_out, right_reg_type_out] = get_dim(*rightFuncType.rhs);

        if (!compat_regs(left_reg_type_in, right_reg_type_in)) {
            throw CompileException("Sorry, I don't know how to BiTensor a " +
                                   leftFuncType.lhs->toString() + " with a " +
                                   rightFuncType.lhs->toString(),
                                   std::move(bitensor.dbg->copy()));
        }

        assert(bitensor.left->getMlirValues().size() == 1
               && bitensor.right->getMlirValues().size() == 1
               && "Expected individual mlir::Values on each side of BiTensor");
        llvm::SmallVector<mlir::Value> captured_funcs{
            bitensor.left->getMlirValues()[0], bitensor.right->getMlirValues()[0]};

        auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
            assert(captures.size() == 2 && args.size() == 1);
            mlir::Value left_func = captures[0];
            mlir::Value right_func = captures[1];
            mlir::Value bundle_in = args[0];

            mlir::ValueRange arg_unpacked;
            if (reg_type_in == RegType::QUBIT) {
                arg_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, bundle_in).getQubits();
            } else if (reg_type_in == RegType::BIT) {
                arg_unpacked = handle.builder.create<qwerty::BitBundleUnpackOp>(loc, bundle_in).getBits();
            }
            assert(arg_unpacked.size() == left_dim_in + right_dim_in
                   && "qubit dimension mismatch in tensoring. how did AST "
                      "typechecking not catch this?");
            llvm::SmallVector<mlir::Value> left_elems, right_elems;
            left_elems.append(arg_unpacked.begin(), arg_unpacked.begin() + left_dim_in);
            right_elems.append(arg_unpacked.begin() + left_dim_in, arg_unpacked.end());

            mlir::Value left_bundle, right_bundle;
            mlir::ValueRange left_bundle_range, right_bundle_range;
            if (!left_elems.empty()) {
                if (left_reg_type_in == RegType::QUBIT) {
                    left_bundle = handle.builder.create<qwerty::QBundlePackOp>(loc, left_elems).getQbundle();
                } else if (left_reg_type_in == RegType::BIT) {
                    left_bundle = handle.builder.create<qwerty::BitBundlePackOp>(loc, left_elems).getBundle();
                } else {
                    assert(0 && "Missing handling of left in bundle case");
                }
                left_bundle_range = left_bundle;
            }
            if (!right_elems.empty()) {
                if (right_reg_type_in == RegType::QUBIT) {
                    right_bundle = handle.builder.create<qwerty::QBundlePackOp>(loc, right_elems).getQbundle();
                } else if (right_reg_type_in == RegType::BIT) {
                    right_bundle = handle.builder.create<qwerty::BitBundlePackOp>(loc, right_elems).getBundle();
                } else {
                    assert(0 && "Missing handling of right in bundle case");
                }
                right_bundle_range = right_bundle;
            }
            mlir::ValueRange left_results = handle.builder.create<qwerty::CallIndirectOp>(
                    loc, left_func, left_bundle_range).getResults();
            mlir::ValueRange right_results = handle.builder.create<qwerty::CallIndirectOp>(
                    loc, right_func, right_bundle_range).getResults();

            mlir::ValueRange left_unpacked, right_unpacked;
            if (!left_results.empty()) {
                assert(left_results.size() == 1 && "I can't unpack multiple bundles at once");
                mlir::Value left_result = left_results[0];
                if (left_reg_type_out == RegType::QUBIT) {
                    left_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, left_result).getQubits();
                } else if (left_reg_type_out == RegType::BIT) {
                    left_unpacked = handle.builder.create<qwerty::BitBundleUnpackOp>(loc, left_result).getBits();
                } else {
                    assert(0 && "Missing handling of left in bundle case");
                }
            }
            if (!right_results.empty()) {
                assert(right_results.size() == 1 && "I can't unpack multiple bundles at once");
                mlir::Value right_result = right_results[0];
                if (right_reg_type_out == RegType::QUBIT) {
                    right_unpacked = handle.builder.create<qwerty::QBundleUnpackOp>(loc, right_result).getQubits();
                } else if (right_reg_type_out == RegType::BIT) {
                    right_unpacked = handle.builder.create<qwerty::BitBundleUnpackOp>(loc, right_result).getBits();
                } else {
                    assert(0 && "Missing handling of right in bundle case");
                }
            }
            llvm::SmallVector<mlir::Value> merged;
            merged.append(left_unpacked.begin(), left_unpacked.end());
            merged.append(right_unpacked.begin(), right_unpacked.end());
            mlir::Value repacked;
            if (reg_type_out == RegType::QUBIT) {
                repacked = handle.builder.create<qwerty::QBundlePackOp>(loc, merged).getQbundle();
            } else if (reg_type_out == RegType::BIT) {
                repacked = handle.builder.create<qwerty::BitBundlePackOp>(loc, merged).getBundle();
            }
            handle.builder.create<qwerty::ReturnOp>(loc, repacked);
        };
        bitensor.getMlirValues() = {createLambda(loc, *funcType, captured_funcs, lambda_contents)};
    } else {
        throw CompileException("Sorry, I don't know how to BiTensor into a "
                               + bitensor.getType().toString(),
                               std::move(bitensor.dbg->copy()));
    }
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) {
    throw CompileException("BroadcastTensor should not reach lowering. This is a "
                           "compiler bug, sorry.",
                           std::move(broadtensor.dbg->copy()));
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, QubitLiteral &lit) {
    mlir::Location loc = lit.dbg->toMlirLoc(handle);
    assert(lit.type.dim->isConstant() && "type variable escaped into mlir lowering!");
    // Speculatively treat this as preparing a state. We will erase the prep op
    // if this is actually in a BasisLiteral
    qwerty::PrimitiveBasis prim_basis = prim_basis_to_qwerty(lit.prim_basis);
    qwerty::Eigenstate eigenstate = eigenstate_to_qwerty(lit.eigenstate);
    uint64_t dim = lit.type.dim->offset;
    lit.getMlirValues() = {handle.builder.create<qwerty::QBundlePrepOp>(
            loc, prim_basis, eigenstate, dim).getResult()};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Phase &phase) {
    phase.phase->visit(ctx, *this);
    phase.value->visit(ctx, *this);
    mlir::Location loc = phase.dbg->toMlirLoc(handle);
    const Type &type = phase.getType();
    const FuncType *funcType;

    assert(phase.phase->getMlirValues().size() == 1
           && "Expected 1 mlir Value for phase");
    assert(phase.value->getMlirValues().size() == 1
           && "Expected 1 mlir Value for value");

    mlir::Value theta = phase.phase->getMlirValues()[0];
    mlir::Value value = phase.value->getMlirValues()[0];

    if (dynamic_cast<const QubitType *>(&type)) {
        phase.getMlirValues() = {handle.builder.create<qwerty::QBundlePhaseOp>(loc, theta, value).getQbundleOut()};
    } else if ((funcType = dynamic_cast<const FuncType *>(&type))
               && dynamic_cast<const QubitType *>(funcType->lhs.get())
               && dynamic_cast<const QubitType *>(funcType->rhs.get())) {
        llvm::SmallVector<mlir::Value> both_captures{value, theta};
        auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
            assert(captures.size() == 2 && args.size() == 1);
            mlir::Value func = captures[0];
            mlir::Value angle = captures[1];
            mlir::Value qbundle_in = args[0];
            mlir::Value res = handle.builder.create<qwerty::CallIndirectOp>(loc, func, qbundle_in).getResults()[0];
            mlir::Value shifted = handle.builder.create<qwerty::QBundlePhaseOp>(loc, angle, res).getQbundleOut();
            handle.builder.create<qwerty::ReturnOp>(loc, shifted);
        };
        phase.getMlirValues() = {createLambda(loc, *funcType, both_captures, lambda_contents)};
    } else {
        throw CompileException("I do not know how to handle a Phase of type " +
                               type.toString(),
                               std::move(phase.dbg->copy()));
    }
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, FloatLiteral &float_) {
    mlir::Location loc = float_.dbg->toMlirLoc(handle);
    mlir::ValueRange stat_vals = wrapStationary(loc, handle.builder.getF64Type(), {}, [&](mlir::ValueRange args) {
        assert(args.empty());
        mlir::Value val =
            handle.builder.create<mlir::arith::ConstantOp>(
                loc, handle.builder.getF64FloatAttr(float_.value)).getResult();
        handle.builder.create<qcirc::CalcYieldOp>(loc, val);
    });
    assert(stat_vals.size() == 1);
    float_.getMlirValues() = {stat_vals[0]};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, FloatNeg &neg) {
    neg.operand->visit(ctx, *this);
    assert(neg.operand->getMlirValues().size() == 1
           && "Expected one MLIR value for FloatNeg operand");
    mlir::Value operand = neg.operand->getMlirValues()[0];
    mlir::Location loc = neg.dbg->toMlirLoc(handle);
    mlir::ValueRange stat_vals = wrapStationary(loc, handle.builder.getF64Type(), operand, [&](mlir::ValueRange args) {
        assert(args.size() == 1);
        mlir::Value operand_arg = args[0];
        mlir::Value val =
            handle.builder.create<mlir::arith::NegFOp>(
                loc, operand_arg).getResult();
        handle.builder.create<qcirc::CalcYieldOp>(loc, val);
    });
    assert(stat_vals.size() == 1);
    neg.getMlirValues() = {stat_vals[0]};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, FloatBinaryOp &bin) {
    bin.left->visit(ctx, *this);
    bin.right->visit(ctx, *this);
    assert(bin.left->getMlirValues().size() == 1
           && bin.right->getMlirValues().size() == 1
           && "Expected one MLIR value for each FloatBinaryOp operand");

    mlir::Location loc = bin.dbg->toMlirLoc(handle);
    mlir::Value left = bin.left->getMlirValues()[0];
    mlir::Value right = bin.right->getMlirValues()[0];


    mlir::ValueRange stat_vals = wrapStationary(
        loc, handle.builder.getF64Type(),
        std::initializer_list<mlir::Value>{left, right},
        [&](mlir::ValueRange args) {
            assert(args.size() == 2);
            mlir::Value left_arg = args[0];
            mlir::Value right_arg = args[1];
            mlir::Value result;
            switch (bin.op) {
            case FLOAT_DIV:
                result = handle.builder.create<mlir::arith::DivFOp>(
                    loc, left_arg, right_arg).getResult();
                break;

            case FLOAT_POW:
                result = handle.builder.create<mlir::math::PowFOp>(
                    loc, left_arg, right_arg).getResult();
                break;

            case FLOAT_MUL:
                result = handle.builder.create<mlir::arith::MulFOp>(
                    loc, left_arg, right_arg).getResult();
                break;

            case FLOAT_ADD:
                result = handle.builder.create<mlir::arith::AddFOp>(
                    loc, left_arg, right_arg).getResult();
                break;

            case FLOAT_MOD:
                result = handle.builder.create<mlir::arith::RemFOp>(
                    loc, left_arg, right_arg).getResult();
                break;

            default:
                assert(0 && "Missing FloatOp handling in FloatBinaryOp lowering");
            }

            handle.builder.create<qcirc::CalcYieldOp>(loc, result);
        });

    assert(stat_vals.size() == 1);
    bin.getMlirValues() = {stat_vals[0]};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) {
    assert(0 && "FloatDimVarExpr should be taken care of in "
                "EvalDimVarExprVisitor, how is it reaching qpu_lowering?");
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, TupleLiteral &tuple) {
    tuple.getMlirValues().clear();
    for (size_t i = 0; i < tuple.elts.size(); i++) {
        tuple.elts[i]->visit(ctx, *this);
        ASTNode::Values &elt_values = tuple.elts[i]->getMlirValues();
        tuple.getMlirValues().append(elt_values.begin(), elt_values.end());
    }
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, BuiltinBasis &std) {
    llvm::SmallVector<qwerty::BasisVectorAttr> vectors;
    assert(std.type.dim->isConstant() && "dimvars snuck into mlir lowering!");
    qwerty::PrimitiveBasis prim_basis = prim_basis_to_qwerty(std.prim_basis);
    uint64_t dim = std.type.dim->offset;
    qwerty::BasisAttr basis = handle.builder.getAttr<qwerty::BasisAttr>(
        handle.builder.getAttr<qwerty::BasisElemAttr>(
            handle.builder.getAttr<qwerty::BuiltinBasisAttr>(prim_basis, dim)));
    ASTNode::BasisValue &std_basis = std.getBasis();
    std_basis.basis = basis;
    std_basis.phases.clear();
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Identity &id) {
    mlir::Location loc = id.dbg->toMlirLoc(handle);
    const Type &type = id.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    assert(funcType
           && dynamic_cast<const QubitType *>(funcType->lhs.get())
           && dynamic_cast<const QubitType *>(funcType->rhs.get())
           && "Identity mapper is not a function from qubits to qubits, huh?");
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(args.size() == 1);
        mlir::Value qbundle_in = args[0];
        mlir::Value id = handle.builder.create<qwerty::QBundleIdentityOp>(loc, qbundle_in).getQbundleOut();
        handle.builder.create<qwerty::ReturnOp>(loc, id);
    };
    id.getMlirValues() = {createLambda(loc, *funcType, mlir::ValueRange(), lambda_contents)};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, BasisTranslation &trans) {
    trans.basis_in->visit(ctx, *this);
    trans.basis_out->visit(ctx, *this);

    mlir::Location loc = trans.dbg->toMlirLoc(handle);
    const Type &type = trans.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    assert(funcType
           && dynamic_cast<const QubitType *>(funcType->lhs.get())
           && dynamic_cast<const QubitType *>(funcType->rhs.get())
           && "Subspace mapper is not a function from qubits to qubits, huh?");
    ASTNode::BasisValue &left_basis = trans.basis_in->getBasis();
    ASTNode::BasisValue &right_basis = trans.basis_out->getBasis();
    llvm::SmallVector<mlir::Value> phase_captures(left_basis.phases.begin(),
                                                  left_basis.phases.end());
    phase_captures.append(right_basis.phases.begin(),
                          right_basis.phases.end());
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(args.size() == 1);
        mlir::Value qbundle_in = args[0];
        mlir::Value mapped = handle.builder.create<qwerty::QBundleBasisTranslationOp>(
                loc, left_basis.basis, right_basis.basis, captures, qbundle_in
            ).getQbundleOut();
        handle.builder.create<qwerty::ReturnOp>(loc, mapped);
    };
    trans.getMlirValues() = {createLambda(loc, *funcType, phase_captures, lambda_contents)};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Discard &discard) {
    mlir::Location loc = discard.dbg->toMlirLoc(handle);
    const Type &type = discard.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    [[maybe_unused]] const TupleType *ret_type;
    assert(funcType
           && dynamic_cast<const QubitType *>(funcType->lhs.get())
           && (ret_type = dynamic_cast<const TupleType *>(funcType->rhs.get()))
           && ret_type->isUnit()
           && "Discard reducer is not a function from qubits to Unit, huh?");
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(captures.empty() && args.size() == 1);
        mlir::Value qbundle_in = args[0];
        handle.builder.create<qwerty::QBundleDiscardOp>(loc, qbundle_in);
        handle.builder.create<qwerty::ReturnOp>(loc, mlir::ValueRange());
    };
    discard.getMlirValues() = {createLambda(loc, *funcType, mlir::ValueRange(), lambda_contents)};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Measure &meas) {
    meas.basis->visit(ctx, *this);
    mlir::Location loc = meas.dbg->toMlirLoc(handle);
    const Type &type = meas.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    assert(funcType
           && dynamic_cast<const QubitType *>(funcType->lhs.get())
           && dynamic_cast<const BitType *>(funcType->rhs.get())
           && "Measure mapper is not a function from qubits to bits, huh?");
    ASTNode::BasisValue &meas_basis = meas.basis->getBasis();
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(captures.empty() && args.size() == 1);
        mlir::Value qbundle_in = args[0];
        mlir::Value measured = handle.builder.create<qwerty::QBundleMeasureOp>(
                loc, meas_basis.basis.deletePhases(), qbundle_in
            ).getBits();
        handle.builder.create<qwerty::ReturnOp>(loc, measured);
    };
    meas.getMlirValues() = {createLambda(loc, *funcType, mlir::ValueRange(), lambda_contents)};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Project &proj) {
    proj.basis->visit(ctx, *this);
    mlir::Location loc = proj.dbg->toMlirLoc(handle);
    const Type &type = proj.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    assert(funcType
           && dynamic_cast<const QubitType *>(funcType->lhs.get())
           && dynamic_cast<const QubitType *>(funcType->rhs.get())
           && "Measure mapper is not a function from qubits to qubits, huh?");
    ASTNode::BasisValue &proj_basis = proj.basis->getBasis();
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(captures.empty() && args.size() == 1);
        mlir::Value qbundle_in = args[0];
        mlir::Value projected = handle.builder.create<qwerty::QBundleProjectOp>(
                loc, proj_basis.basis.deletePhases(), qbundle_in
            ).getQbundleOut();
        handle.builder.create<qwerty::ReturnOp>(loc, projected);
    };
    proj.getMlirValues() = {createLambda(loc, *funcType, mlir::ValueRange(), lambda_contents)};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Flip &flip) {
    flip.basis->visit(ctx, *this);
    mlir::Location loc = flip.dbg->toMlirLoc(handle);
    const Type &type = flip.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    assert(funcType && "Flip is not a function, how?");
    ASTNode::BasisValue &flip_basis = flip.basis->getBasis();
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(args.size() == 1);
        mlir::Value qbundle_in = args[0];
        mlir::Value flipped = handle.builder.create<qwerty::QBundleFlipOp>(
                loc, flip_basis.basis, captures, qbundle_in
            ).getQbundleOut();
        handle.builder.create<qwerty::ReturnOp>(loc, flipped);
    };
    flip.getMlirValues() = {createLambda(loc, *funcType, flip_basis.phases, lambda_contents)};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Rotate &rot) {
    rot.theta->visit(ctx, *this);
    rot.basis->visit(ctx, *this);
    mlir::Location loc = rot.dbg->toMlirLoc(handle);
    const Type &type = rot.getType();
    const FuncType *funcType = dynamic_cast<const FuncType *>(&type);
    assert(funcType
           && "Rotate is not a function, huh?");
    assert(rot.theta->getMlirValues().size() == 1
           && "Wrong number of Values for theta of Rotate");
    ASTNode::BasisValue &rot_basis = rot.basis->getBasis();
    mlir::Value theta = rot.theta->getMlirValues()[0];
    auto lambda_contents = [&](mlir::ValueRange captures, mlir::ValueRange args) {
        assert(captures.size() == 1 && args.size() == 1);
        mlir::Value angle = captures[0];
        mlir::Value qbundle_in = args[0];
        mlir::Value rotated = handle.builder.create<qwerty::QBundleRotateOp>(
                loc, rot_basis.basis.deletePhases(), angle, qbundle_in
            ).getQbundleOut();
        handle.builder.create<qwerty::ReturnOp>(loc, rotated);
    };
    rot.getMlirValues() = {createLambda(loc, *funcType, theta, lambda_contents)};
    return true;
}

// already_visited avoids re-visiting the angle for Phase nodes when we
// encounter a BroadcastTensor
void QpuLoweringVisitor::walkBasisList(
        ASTNode *node, ASTVisitContext &ctx, bool already_visited,
        qwerty::PrimitiveBasis &prim_basis_out, mlir::Value &theta_out,
        llvm::APInt &eigenbits_out) {
    TupleLiteral *tup;
    if (Phase *phase = dynamic_cast<Phase *>(node)) {
        phase->phase->visit(ctx, *this);
        assert(phase->phase->getMlirValues().size() == 1
               && "Phase of phase has wrong number of MLIR values");
        mlir::Value next_theta = phase->phase->getMlirValues()[0];
        if (!theta_out) {
            theta_out = next_theta;
        } else {
            mlir::Location loc = phase->dbg->toMlirLoc(handle);
            mlir::ValueRange stat_vals = wrapStationary(loc,
                    handle.builder.getF64Type(),
                    std::initializer_list<mlir::Value>{theta_out, next_theta},
                    [&](mlir::ValueRange args) {
                assert(args.size() == 2);
                mlir::Value theta_out_arg = args[0];
                mlir::Value next_theta_arg = args[1];
                mlir::Value val =
                    handle.builder.create<mlir::arith::AddFOp>(
                        loc, theta_out_arg, next_theta_arg).getResult();
                handle.builder.create<qcirc::CalcYieldOp>(loc, val);
            });
            assert(stat_vals.size() == 1);
            theta_out = stat_vals[0];
        }
        walkBasisList(phase->value.get(), ctx, /*already_visited=*/false,
                      prim_basis_out, theta_out, eigenbits_out);
    } else if (QubitLiteral *lit = dynamic_cast<QubitLiteral *>(node)) {
        qwerty::PrimitiveBasis prim_basis = prim_basis_to_qwerty(lit->prim_basis);
        if (prim_basis_out == (qwerty::PrimitiveBasis)-1) {
            prim_basis_out = prim_basis;
        } else if (prim_basis_out != prim_basis) {
            throw CompileException("Mismatch in PrimitiveBasiss for qubits of basis "
                                   "literal. Earlier I saw "
                                   + qwerty::stringifyPrimitiveBasis(prim_basis_out).str()
                                   + " but now I am seeing "
                                   + qwerty::stringifyPrimitiveBasis(prim_basis).str(),
                                   std::move(lit->dbg->copy()));
        }

        assert(lit->type.dim->isConstant()
               && "typevars snuck into basis vector lowering");
        size_t dim = lit->type.dim->offset;
        eigenbits_out = eigenbits_out.zext(eigenbits_out.getBitWidth() + dim);
        eigenbits_out = eigenbits_out << dim;
        if (lit->eigenstate == MINUS) {
            eigenbits_out.setBits(0, dim);
        }
    } else if (BiTensor *bi_tensor = dynamic_cast<BiTensor *>(node)) {
        walkBasisList(bi_tensor->left.get(), ctx, /*already_visited=*/false,
                      prim_basis_out, theta_out, eigenbits_out);
        walkBasisList(bi_tensor->right.get(), ctx, /*already_visited=*/false,
                      prim_basis_out, theta_out, eigenbits_out);
    } else if (BroadcastTensor *broad_tensor = dynamic_cast<BroadcastTensor *>(node)) {
        assert(broad_tensor->factor->isConstant() && "Dimvar snuck into basis list lowering!");
        DimVarValue factor = broad_tensor->factor->offset;
        for (DimVarValue i = 0; i < factor; i++) {
            walkBasisList(broad_tensor->value.get(), ctx, already_visited || i > 0,
                          prim_basis_out, theta_out, eigenbits_out);
        }
    } else if ((tup = dynamic_cast<TupleLiteral *>(node)) && tup->isUnit()) {
        // Don't do anything
    } else {
        throw CompileException("Unknown node in basis vector: "
                               + node->label(),
                               std::move(node->dbg->copy()));
    }
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, BasisLiteral &lit) {
    assert(!lit.elts.empty() && "Empty BasisLiteral makes no sense, shouldn't "
                                "typechecking catch this?");
    llvm::SmallVector<mlir::Value> thetas;
    llvm::SmallVector<qwerty::BasisVectorAttr> vectors;
    for (size_t i = 0; i < lit.elts.size(); i++) {
        qwerty::PrimitiveBasis prim_basis = (qwerty::PrimitiveBasis)-1;
        mlir::Value theta;
        llvm::APInt eigenbits(/*numBits=*/0UL, /*val=*/0UL, /*isSigned=*/false);
        walkBasisList(lit.elts[i].get(), ctx, /*already_visited=*/false,
                      prim_basis, theta, eigenbits);

        const QubitType *elt_type = dynamic_cast<const QubitType *>(&lit.elts[i]->getType());
        assert(elt_type && "Type of basis element is not qubit, how?");
        assert(elt_type->dim->isConstant() && "Dimvars snuck into basis elements");
        DimVarValue n_qubits = elt_type->dim->offset;

        qwerty::BasisVectorAttr vec_attr = handle.builder.getAttr<qwerty::BasisVectorAttr>(
                prim_basis, eigenbits, n_qubits, !!theta);
        vectors.push_back(vec_attr);
        if (theta) {
            thetas.push_back(theta);
        }
    }

    qwerty::BasisAttr basis = handle.builder.getAttr<qwerty::BasisAttr>(
            handle.builder.getAttr<qwerty::BasisElemAttr>(
                handle.builder.getAttr<qwerty::BasisVectorListAttr>(vectors)));
    ASTNode::BasisValue &lit_basis = lit.getBasis();
    lit_basis.basis = basis;
    lit_basis.phases = thetas;
    return true;
}

void QpuLoweringVisitor::walkSuperposOperandHelper(
        ASTNode &node,
        qwerty::PrimitiveBasis &prim_basis_out,
        double &theta_out,
        llvm::APInt &eigenbits_out,
        llvm::SmallVector<qwerty::BasisVectorAttr> &vecs_out) {
    TupleLiteral *tup;
    if (Phase *phase = dynamic_cast<Phase *>(&node)) {
        FloatLiteral *next_theta = dynamic_cast<FloatLiteral*>(phase->phase.get());

        if (!next_theta) {
            throw CompileException(
                "Phases in a superposition literal must be constant",
                std::move(node.dbg->copy()));
        }

        theta_out += next_theta->value;
        walkSuperposOperandHelper(
            *phase->value, prim_basis_out, theta_out, eigenbits_out,
            vecs_out);
    } else if (QubitLiteral *lit = dynamic_cast<QubitLiteral *>(&node)) {
        qwerty::PrimitiveBasis prim_basis = prim_basis_to_qwerty(lit->prim_basis);
        if (prim_basis_out == (qwerty::PrimitiveBasis)-1) {
            prim_basis_out = prim_basis;
        } else if (prim_basis_out != prim_basis) {
            size_t n_qubits = eigenbits_out.getBitWidth();
            qwerty::BasisVectorAttr vec_attr =
                handle.builder.getAttr<qwerty::BasisVectorAttr>(
                    prim_basis_out, eigenbits_out,
                    n_qubits, /*hasPhase=*/false);
            llvm::APInt eigenbits(/*numBits=*/0UL, /*val=*/0UL, /*isSigned=*/false);
            vecs_out.push_back(vec_attr);
            prim_basis_out = prim_basis;
            eigenbits_out = eigenbits;
        }

        assert(lit->type.dim->isConstant()
               && "typevars snuck into basis vector lowering");
        size_t dim = lit->type.dim->offset;
        eigenbits_out = eigenbits_out.zext(eigenbits_out.getBitWidth() + dim);
        eigenbits_out <<= dim;
        if (lit->eigenstate == MINUS) {
            eigenbits_out.setBits(0, dim);
        }
    } else if (BiTensor *bi_tensor = dynamic_cast<BiTensor *>(&node)) {
        walkSuperposOperandHelper(
            *bi_tensor->left, prim_basis_out, theta_out, eigenbits_out,
            vecs_out);
        walkSuperposOperandHelper(
            *bi_tensor->right, prim_basis_out, theta_out, eigenbits_out,
            vecs_out);
    } else if (BroadcastTensor *broad_tensor = dynamic_cast<BroadcastTensor *>(&node)) {
        assert(broad_tensor->factor->isConstant() && "Dimvar snuck into basis list lowering!");
        DimVarValue factor = broad_tensor->factor->offset;
        for (DimVarValue i = 0; i < factor; i++) {
            walkSuperposOperandHelper(
                *broad_tensor->value, prim_basis_out, theta_out, eigenbits_out,
                vecs_out);
        }
    } else if ((tup = dynamic_cast<TupleLiteral *>(&node)) && tup->isUnit()) {
        // Don't do anything
    } else {
        throw CompileException("Unknown node in superposition literal: "
                               + node.label(),
                               std::move(node.dbg->copy()));
    }
}

void QpuLoweringVisitor::walkSuperposOperand(
        ASTNode &node,
        double &theta_out,
        llvm::SmallVector<qwerty::BasisVectorAttr> &vecs_out) {
    theta_out = 0;
    qwerty::PrimitiveBasis prim_basis = (qwerty::PrimitiveBasis)-1;
    llvm::APInt eigenbits(/*numBits=*/0UL, /*val=*/0UL, /*isSigned=*/false);
    walkSuperposOperandHelper(
        node, prim_basis, theta_out, eigenbits, vecs_out);

    // This could be 0 if the last operand is ()
    if (eigenbits.getBitWidth()) {
        size_t n_qubits = eigenbits.getBitWidth();
        qwerty::BasisVectorAttr vec_attr =
            handle.builder.getAttr<qwerty::BasisVectorAttr>(
                prim_basis, eigenbits, n_qubits, /*hasPhase=*/false);
        vecs_out.push_back(vec_attr);
    }
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, SuperposLiteral &lit) {
    mlir::Location loc = lit.dbg->toMlirLoc(handle);
    assert(!lit.pairs.empty()
           && "Empty Superposition makes no sense, shouldn't typechecking "
              "catch this?");
    llvm::SmallVector<qwerty::SuperposElemAttr> opperands;
    for (size_t i = 0; i < lit.pairs.size(); i++) {
        double prob_val = lit.pairs[i].first;
        mlir::FloatAttr prob = handle.builder.getF64FloatAttr(prob_val);

        ASTNode &elt = *lit.pairs[i].second;
        double theta;
        llvm::SmallVector<qwerty::BasisVectorAttr> vectors;
        walkSuperposOperand(elt, theta, vectors);
        assert(!vectors.empty() && "Empty superpos element. Shouldn't "
                                   "typechecking catch this?");

        mlir::FloatAttr floatTheta = handle.builder.getF64FloatAttr(theta);
        qwerty::SuperposElemAttr superpos_elem =
            handle.builder.getAttr<qwerty::SuperposElemAttr>(
                prob, floatTheta, vectors);
        opperands.push_back(superpos_elem);
    }

    qwerty::SuperposAttr superpos_attr =
        handle.builder.getAttr<qwerty::SuperposAttr>(opperands);
    lit.getMlirValues() = {
        handle.builder.create<qwerty::SuperposOp>(
            loc, superpos_attr).getQbundle()};
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Conditional &cond) {
    mlir::Location loc = cond.dbg->toMlirLoc(handle);
    cond.if_expr->visit(ctx, *this);

    assert(cond.if_expr->getMlirValues().size() == 1
            && "Condition's test should be one MLIR value");

    mlir::Value if_expr = cond.if_expr->getMlirValues()[0];
    // if_expr is a bitbundle, so we unpack it and extract the bit
    mlir::ValueRange bits = handle.builder.create<qwerty::BitBundleUnpackOp>(loc, if_expr).getBits();
    assert(bits.size() == 1 && "Found more than one bit for conditional's test.");
    mlir::scf::IfOp if_op = handle.builder.create<mlir::scf::IfOp>(loc, cond.getType().toMlirType(handle), bits[0], true);
    mlir::Block *then_block = if_op.thenBlock();
    mlir::Block *else_block = if_op.elseBlock();

    {
        mlir::OpBuilder::InsertionGuard guard(handle.builder);
        handle.builder.setInsertionPointToStart(then_block);
        cond.then_expr->visit(ctx, *this);
        handle.builder.create<mlir::scf::YieldOp>(loc, cond.then_expr->getMlirValues());

        handle.builder.setInsertionPointToStart(else_block);
        cond.else_expr->visit(ctx, *this);
        handle.builder.create<mlir::scf::YieldOp>(loc, cond.else_expr->getMlirValues());
    }

    cond.getMlirValues() = if_op.getResults();
    return true;
}

bool QpuLoweringVisitor::visit(ASTVisitContext &ctx, Slice &slice) {
    slice.val->visit(ctx, *this);
    ASTNode::Values &val_vals = slice.val->getMlirValues();
    assert(slice.lower->isConstant()
           && slice.upper->isConstant()
           && "Dimvars snuck into slice lowering");
    assert(slice.lower->offset < (ssize_t)val_vals.size()
           && slice.upper->offset <= (ssize_t)val_vals.size()
           && "Slice indices out of range. Why didn't typechecking catch this?");

    ASTNode::Values &slice_vals = slice.getMlirValues();
    slice_vals.clear();
    slice_vals.append(val_vals.begin() + slice.lower->offset,
                      val_vals.begin() + slice.upper->offset);
    return true;
}
