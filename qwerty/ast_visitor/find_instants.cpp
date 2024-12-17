#include "ast.hpp"
#include "ast_visitor.hpp"

void FindInstantiationsVisitor::materializeInstantiation(
        std::string capture_name,
        const std::vector<DimVarValue> &instance_vals,
        DebugInfo &dbg) {
    // Unfortunate linear search
    ssize_t i = -1;
    for (size_t j = 0; j < kernel.capture_names.size(); j++) {
        if (kernel.capture_names[j] == capture_name) {
            i = (ssize_t)j;
            break;
        }
    }
    assert(i >= 0 && "Undefined variable. Type checking bug?");
    assert(!instance_vals.empty() && "Empty list of instance vals");

    if (kernel.capture_freevars[i] != instance_vals.size()) {
        throw TypeException("Can't instantiate " + kernel.capture_names[i]
                            + " since it has "
                            + std::to_string(kernel.capture_freevars[i])
                            + " declared free variables, not"
                            + std::to_string(instance_vals.size()),
                            std::move(dbg.copy()));
    }
    if (!kernel.capture_objs[i]->needsExplicitDimvars()) {
        throw TypeException("Why instantiate " + kernel.capture_names[i]
                            + "? It does not need any type variables",
                            std::move(dbg.copy()));
    }
    if (kernel.capture_instances[i]->count(instance_vals)) {
        // Nothing to do, we should be good
    } else {
        std::unique_ptr<HybridObj> copy = kernel.capture_objs[i]->copyHybrid();
        std::vector<DimVarValue> vals(instance_vals.begin(), instance_vals.end());
        copy->instantiateWithExplicitDimvars(vals);
        kernel.capture_instances[i]->emplace(instance_vals, std::move(copy));
    }
}

bool FindInstantiationsVisitor::visit(ASTVisitContext &ctx, Instantiate &instant) {
    std::vector<DimVarValue> instance_vals;
    instance_vals.reserve(instant.instance_vals.size());

    for (std::unique_ptr<DimVarExpr> &val : instant.instance_vals) {
        assert(val->isConstant() && "Instantiate val is not constant");
        instance_vals.push_back(val->offset);
    }

    if (Variable *var = dynamic_cast<Variable *>(instant.var.get())) {
        materializeInstantiation(var->name, instance_vals, *instant.dbg);
    } else if (EmbedClassical *embed = dynamic_cast<EmbedClassical *>(instant.var.get())) {
        materializeInstantiation(embed->name, instance_vals, *instant.dbg);
        if (!embed->operand_name.empty()) {
            materializeInstantiation(embed->operand_name, instance_vals, *instant.dbg);
        }
    } else {
        assert(0 && "Instantiate val is not a variable. How did this pass typechecking?");
    }

    return true;
}

