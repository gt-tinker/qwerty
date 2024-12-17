#include <fstream>
#include "ast.hpp"

namespace {
// Return the dimension of a bit[N], e.g., this part:
//     bit[N+M+3]
//         ^^^^^
// (Return nullptr if this is not a bit array type.)
DimVarExpr *extractBitDim(Type *type) {
    BitType *bit;
    BroadcastType *broad;

    if ((bit = dynamic_cast<BitType *>(type))) {
        return bit->dim.get();
    } else if ((broad = dynamic_cast<BroadcastType *>(type))
               || ((bit = dynamic_cast<BitType *>(broad->elem_type.get()))
                    && bit->dim->isConstant()
                    && bit->dim->offset == 1)) {
        return broad->factor.get();
    } else {
        return nullptr;
    }
}
} // namespace

bool qwerty_debug = false;

///////////////////////// AST TRAVERSAL /////////////////////////

void ASTNode::walk(ASTVisitContext &ctx, ASTVisitor &visitor) {
    Traversal traversal = visitor.traversal();
    if (traversal == Traversal::PREORDER
            || traversal == Traversal::PREPOSTORDER) {
        if (!this->visit(ctx, visitor)) {
            // Transfer control over to the new version of me (provided he exists)
            if (ctx.ptr) {
                ctx.ptr->walk(ctx, visitor);
            }
            return;
        }
    }

    if (traversal != Traversal::CUSTOM) {
        for (std::pair<std::string, std::unique_ptr<ASTNode> &> &label_child : children()) {
            std::string label = label_child.first;
            std::unique_ptr<ASTNode> &child = label_child.second;
            ASTVisitContext ctx(*this, label, child);
            child->walk(ctx, visitor);
        }
        if (traversal == Traversal::POSTORDER
                || traversal == Traversal::PREPOSTORDER) {
            this->visit(ctx, visitor);
        }
    } else { // Do Custom
        this->visit(ctx, visitor);
    }
}

void ASTNode::walk(ASTVisitor &visitor) {
    visitor.init(*this);
    std::unique_ptr<ASTNode> dummy = nullptr;
    ASTVisitContext ctx(*this, "root", dummy);
    walk(ctx, visitor);
    visitor.finish();
}

std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> BasisLiteral::children() {
    std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> kids;
    kids.reserve(elts.size());
    for (size_t i = 0; i < elts.size(); i++) {
        std::unique_ptr<ASTNode> &elt = elts[i];
        kids.push_back({std::to_string(i), elt});
    }
    return kids;
}

std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> TupleLiteral::children() {
    std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> kids;
    kids.reserve(elts.size());
    for (size_t i = 0; i < elts.size(); i++) {
        std::unique_ptr<ASTNode> &elt = elts[i];
        kids.push_back({std::to_string(i), elt});
    }
    return kids;
}

std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> Kernel::children() {
    std::vector<std::pair<std::string, std::unique_ptr<ASTNode> &>> kids;
    kids.reserve(body.size());
    for (size_t i = 0; i < body.size(); i++) {
        std::unique_ptr<ASTNode> &stmt = body[i];
        kids.push_back({std::to_string(i), stmt});
    }
    return kids;
}

///////////////////////// SPANS /////////////////////////

void VeclistSpan::dump(std::ostream &os) const {
    os << prim_basis_name(prim_basis) << "{";

    // Print vectors in a deterministic order
    auto apint_compare = [](llvm::APInt a, llvm::APInt b) { return a.ult(b); };
    std::set<llvm::APInt, decltype(apint_compare)> sorted(vecs.begin(),
                                                          vecs.end(),
                                                          apint_compare);
    bool first = true;
    for (const llvm::APInt &v : sorted) {
        if (first) {
            first = false;
        } else {
            os << ",";
        }
        std::string bits(v.getBitWidth(), '0');
        for (size_t i = 0; i < bits.size(); i++) {
            if (v[bits.size()-1-i]) {
                bits[i] = '1';
            }
        }
        os << bits;
    }

    os << "}";
}

std::ostream &operator<<(std::ostream &os, const Span &sp) {
    sp.dump(os);
    return os;
}

std::ostream &operator<<(std::ostream &os, const SpanList &span) {
    bool first = true;
    for (const std::unique_ptr<Span> &sp : span.list) {
        if (first) {
            first = false;
        } else {
            os << " + ";
        }
        os << *sp;
    }
    return os;
}

namespace {
void factorFullOutOfFullSpan(SpanList &span, size_t i, size_t n) {
    FullSpan &fsp = static_cast<FullSpan &>(*span.list[i]);
    assert(fsp.n_qubits > n && "full span too small");
    size_t delta = fsp.n_qubits - n;
    fsp.n_qubits = n;
    span.list.insert(span.list.begin()+i+1,
                     std::make_unique<FullSpan>(delta));
}

// Let |bv⟩ denote an l-qubit standard computational basis state. ℋ denotes a
// 2D hilbert space (i.e., the state space of 1 qubit).
//
// Lemma 1: {bv1,bv2,...,bvm_1} = ⋃_(k=0)^(2^n-1) {|k⟩|bv_j'⟩ : 1 <= j <= m_2}
//          if and only if
//          span({bv1,bv2,...,bvm_1}) = ℋ^⊗n ⊗  span({bv1',bv2',...,bvm_2'})
// Proof:
// (==>) Definition of the tensor product ⊗ on vector spaces.
// (<==) Similar.
//
// Corollary 2: If 2^n does not divide m_1, then there are no bv_j' such that
//              span({bv1,bv2,...,bvm_1}) = ℋ^⊗n ⊗  span({bv1',bv2',...,bvm_2'})
// Proof (by contrapositive):
//   Suppose span({bv1,bv2,...,bvm_1}) = ℋ^⊗n ⊗  span({bv1',bv2',...,bvm_2'}).
//   By Lemma 1, {bv1,bv2,...,bvm_1} = ⋃_(k=0)^(2^n-1) {|k⟩|bv_j'⟩ : 1 <= j <= m_2}.
//   Clearly, |⋃_(k=0)^(2^n-1) {|k⟩|bv_j'⟩ : 1 <= j <= m_2}| = 2^n * m_2.
//   Thus 2^n * m_2 = m_1, i.e., m_1 is a mulitple of 2^n. Qed.
//
// Algorithm 3 (Factoring a full span out of a veclist span):
//   Inputs: n and {bv1,bv2,...,bvm_1}
//   Outputs: {bv1',bv2',...,bvm_2'} on success
//   Steps:
//     1. If lower n bits of m_1 are not clear, fail (Corollary 2)
//     2. Scan bvi for all possible basis states (can use an array of size 2^n
//        initialized to false). If there are any missing, fail
//     3. Create a mapping initialized as {bvj' -> 0}. Iterate through all bvi,
//        incrementing the counter for its suffix bvj'.
//     4. If any counters in the mapping are less than 2^n, fail
//     5. Success: convert the keys of the {bvj' -> 2^n} mapping into
//        VeclistSpan and return it. Use the PrimitiveBasis of the original {bvi}
bool factorFullOutOfVeclist(SpanList &offender, size_t i, size_t n) {
    VeclistSpan *vsp = dynamic_cast<VeclistSpan *>(offender.list[i].get());
    assert(vsp && "Non-veclist passed to factorFullOutOfVeclist()");
    size_t n_qubits = vsp->getNumQubits();
    assert(n_qubits > n && "Veclist too small for prefix");

    if (bits_trailing_zeros(vsp->vecs.size()) < n) {
        return false; // Corollary 2
    }

    size_t two_to_the_n = 1ULL << n;
    llvm::SmallVector<bool> prefixes_seen(two_to_the_n, false);
    for (const llvm::APInt &vec : vsp->vecs) {
        size_t prefix = vec.extractBitsAsZExtValue(n, n_qubits - n);
        prefixes_seen[prefix] = true;
    }

    for (bool seen : prefixes_seen) {
        if (!seen) {
            return false; // Step 2 of Algorithm 3 above
        }
    }

    llvm::DenseMap<llvm::APInt, size_t> suffixes_seen;
    for (const llvm::APInt &vec : vsp->vecs) {
        llvm::APInt suffix = vec.trunc(n_qubits - n);
        if (suffixes_seen.count(suffix)) {
            suffixes_seen[suffix]++;
        } else {
            suffixes_seen.insert({suffix, 1});
        }
    }

    std::unique_ptr<FullSpan> new_full = std::make_unique<FullSpan>(n);
    std::unique_ptr<VeclistSpan> new_veclist = std::make_unique<VeclistSpan>(vsp->prim_basis);
    for (const auto& [suffix, count] : suffixes_seen) {
        if (count < two_to_the_n) {
            return false;
        }
        new_veclist->vecs.insert(suffix);
    }

    offender.list[i] = std::move(new_full);
    offender.list.insert(offender.list.begin()+(i+1), std::move(new_veclist));
    return true;
}

bool factorVeclistOutOfVeclist(SpanList &offender, size_t i, VeclistSpan &desired) {
    VeclistSpan *vsp = dynamic_cast<VeclistSpan *>(offender.list[i].get());
    assert(vsp && "Non-veclist passed to factorVeclistOutOfVeclist()");
    size_t n = desired.getNumQubits();
    size_t n_qubits = vsp->getNumQubits();
    assert(n_qubits > n && "Veclist too small for veclist prefix");

    if (vsp->vecs.size() % desired.vecs.size()) {
        return false; // Generalized Corollary 2 above
    }

    std::unordered_set<llvm::APInt> prefixes_seen;
    for (const llvm::APInt &vec : vsp->vecs) {
        llvm::APInt prefix = vec.lshr(n_qubits - n).trunc(n);
        // Needed in this generalization: what if we see a prefix that couldn't
        // have been produced from the factoring we want?
        if (!desired.vecs.count(prefix)) {
            return false;
        }
        prefixes_seen.insert(prefix);
    }

    for (const llvm::APInt &prefix : desired.vecs) {
        if (!prefixes_seen.count(prefix)) {
            return false; // Generalized step 2 of Algorithm 3 above
        }
    }

    llvm::DenseMap<llvm::APInt, size_t> suffixes_seen;
    for (const llvm::APInt &vec : vsp->vecs) {
        llvm::APInt suffix = vec.trunc(n_qubits - n);
        if (suffixes_seen.count(suffix)) {
            suffixes_seen[suffix]++;
        } else {
            suffixes_seen.insert({suffix, 1});
        }
    }

    std::unique_ptr<VeclistSpan> new_veclist = std::make_unique<VeclistSpan>(vsp->prim_basis);
    for (const auto& [suffix, count] : suffixes_seen) {
        if (count < desired.vecs.size()) {
            return false;
        }
        new_veclist->vecs.insert(suffix);
    }

    offender.list[i] = desired.copy();
    offender.list.insert(offender.list.begin()+(i+1), std::move(new_veclist));
    return true;
}

} // namespace

bool operator==(const SpanList &left, const SpanList &right) {
    SpanList lhs, rhs;
    lhs.append(left);
    rhs.append(right);

    size_t i;
    for (i = 0; i < lhs.list.size() && i < rhs.list.size(); i++) {
        Span *lsp = lhs.list[i].get();
        Span *rsp = rhs.list[i].get();
        FullSpan *lfsp = dynamic_cast<FullSpan *>(lsp);
        FullSpan *rfsp = dynamic_cast<FullSpan *>(rsp);
        VeclistSpan *lvsp = dynamic_cast<VeclistSpan *>(lsp);
        VeclistSpan *rvsp = dynamic_cast<VeclistSpan *>(rsp);

        assert(lsp->getNumQubits() && rsp->getNumQubits()
               && "Number of span qubits cannot be zero, bug");

        if (lsp->getNumQubits() == rsp->getNumQubits()) {
            if (lsp->fullySpans() && rsp->fullySpans()) {
                // We match already, keep going
            } else if (lvsp && rvsp && lvsp->vecs == rvsp->vecs
                       && lvsp->prim_basis == rvsp->prim_basis) {
                // We match already, keep going
            } else {
                return false;
            }
        } else {
            bool lbigger = lsp->getNumQubits() > rsp->getNumQubits();
            SpanList &bigger = lbigger? lhs : rhs;
            size_t n_small = lbigger? rsp->getNumQubits() : lsp->getNumQubits();
            FullSpan *fbig = lbigger? lfsp : rfsp;
            FullSpan *fsmall = lbigger? rfsp : lfsp;
            VeclistSpan *vbig = lbigger? lvsp : rvsp;
            VeclistSpan *vsmall = lbigger? rvsp : lvsp;

            if (fbig && fsmall) {
                factorFullOutOfFullSpan(bigger, i, n_small);
            } else if (fbig && vsmall && vsmall->fullySpans()) {
                factorFullOutOfFullSpan(bigger, i, n_small);
            } else if (vbig && fsmall) {
                if (!factorFullOutOfVeclist(bigger, i, n_small)) {
                    return false;
                }
            } else if (vbig && vsmall && vsmall->fullySpans()) {
                if (!factorFullOutOfVeclist(bigger, i, n_small)) {
                    return false;
                }
            } else if (vbig && vsmall && vbig->prim_basis == vsmall->prim_basis) {
                if (!factorVeclistOutOfVeclist(bigger, i, *vsmall)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    if (i < lhs.list.size() || i < rhs.list.size()) {
        return false;
    }

    return true;
}

///////////////////////// TYPE EXPRESSION ARITHMETIC /////////////////////////

DimVarExpr &DimVarExpr::operator+=(const DimVarExpr &other) {
    for (const std::pair<DimVar, DimVarValue> dv_val : other.dimvars) {
        if (!dimvars.count(dv_val.first)) {
            dimvars[dv_val.first] = dv_val.second;
        } else {
            dimvars[dv_val.first] += dv_val.second;
        }
    }
    offset += other.offset;
    return *this;
}

DimVarExpr &DimVarExpr::operator-=(const DimVarExpr &other) {
    for (const std::pair<DimVar, DimVarValue> dv_val : other.dimvars) {
        if (!dimvars.count(dv_val.first)) {
            dimvars[dv_val.first] = -dv_val.second;
        } else {
            dimvars[dv_val.first] -= dv_val.second;
        }
    }
    offset -= other.offset;
    return *this;
}

DimVarExpr &DimVarExpr::operator*=(const DimVarExpr &other) {
    assert((isConstant() || other.isConstant())
           && "Constant expressions must be linear");

    if (isConstant()) {
        for (const std::pair<DimVar, DimVarValue> dv_val : other.dimvars) {
            dimvars[dv_val.first] += offset*dv_val.second;
        }
        offset *= other.offset;
    } else {
        for (const std::pair<DimVar, DimVarValue> dv_val : dimvars) {
            dimvars[dv_val.first] *= other.offset;
        }
    }
    return *this;
}

///////////////////////// COMPILATION /////////////////////////

HybridObj::Hash Bits::getHash() const {
    std::string bits_str;
    bits_str.reserve(bits.size());
    for (bool bit : bits) {
        bits_str.push_back('0' + bit);
    }
    return "Bits:" + bits_str;
}

void Kernel::erase() {
    funcOp_name.clear();
    if (funcOp) {
        funcOp.erase();
    }
    for (size_t i = 0; i < capture_objs.size(); i++) {
        capture_objs[i]->eraseIfPrivate();
    }
}

bool Kernel::needsRecompile(std::vector<HybridObj *> &provided_captures) const {
    if (provided_captures.size() != capture_objs.size()) {
        // This is going to be a type checking error, but we'll let
        // re-compilation surface that
        return true;
    }
    for (size_t i = 0; i < provided_captures.size(); i++) {
        if (provided_captures[i]->getHash() != capture_objs[i]->getHash()) {
            return true;
        }
    }
    return false;
}

void Kernel::compile(MlirHandle &handle) {
    for (size_t i = 0; i < capture_objs.size(); i++) {
        if (capture_instances[i]->empty()) {
            std::string captured_funcOp_name =
                    getFuncOpName() + "__cap" + std::to_string(i);
            capture_objs[i]->compileIfNeeded(handle, captured_funcOp_name, true);
        } else {
            for (auto &kv : *capture_instances[i]) {
                std::ostringstream ss;
                for (const DimVarValue dvv : kv.first) {
                    ss << "_";
                    ss << dvv;
                }
                std::string instantiated_funcOp_name =
                        getFuncOpName() + "__cap" + std::to_string(i)
                        + "__instantiate" + ss.str();
                kv.second->compileIfNeeded(handle, instantiated_funcOp_name, true);
            }
        }
    }

    // We're about to add a new FuncOp, so we'll need to re-JIT after this
    handle.invalidate_jit();
}

///////////////////////// STRING FORMATTING /////////////////////////

std::string DimVarExpr::toString() const {
    std::string sum;
    for (const std::pair<DimVar, DimVarValue> dv_val : dimvars) {
        if (!sum.empty()) {
            sum += "+";
        }
        if (dv_val.second != 1) {
            sum += std::to_string(dv_val.second) + "*";
        }
        sum += dv_val.first;
    }
    if (offset || dimvars.empty()) {
        if (!sum.empty()) {
            sum += "+";
        }
        sum += std::to_string(offset);
    }
    return sum;
}

std::string FuncType::toString() const {
    return "(" + lhs->toString() + (is_rev? "—ʳᵉᵛ→" : "→")
           + rhs->toString() + ")";
}

std::string TupleType::toString() const {
    assert(types.size() != 1 && "A tuple of size 1 is redundant");

    if (types.empty()) {
        return "Unit";
    }

    std::string inner_types;
    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            inner_types += "✕";
        }
        inner_types += types[i]->toString();
    }

    return "(" + inner_types + ")";
}

std::vector<std::string> QubitLiteral::getAdditionalMetadata() const {
    return {"PrimitiveBasis: " + prim_basis_name(prim_basis),
            "Eigenstate: " + eigenstate_name(eigenstate)};
}

std::vector<std::string> BuiltinBasis::getAdditionalMetadata() const {
    return {"PrimitiveBasis: " + prim_basis_name(prim_basis)};
}

///////////////////////// REVERSIBILITY /////////////////////////

std::unique_ptr<Type> Type::operator+(const Type &type) const {
    const TupleType *lhs_tuple = dynamic_cast<const TupleType *>(this);
    const TupleType *rhs_tuple = dynamic_cast<const TupleType *>(&type);
    if (lhs_tuple && rhs_tuple) {
        return std::move(*lhs_tuple + *rhs_tuple);
    } else if (lhs_tuple) {
        return std::move(*lhs_tuple + type);
    } else if (rhs_tuple) {
        return std::move(type + *rhs_tuple);
    } else {
        std::vector<std::unique_ptr<Type>> types(2);
        types[0] = std::move(copy());
        types[1] = std::move(type.copy());
        return std::move(std::make_unique<TupleType>(std::move(types)));
    }
}

bool TupleType::isCanonical() const {
    BroadcastType *left_broad, *right_broad;
    for (size_t i = 0; i+1 < types.size(); i++) {
        if ((dynamic_cast<QubitType *>(types[i].get())
             && dynamic_cast<QubitType *>(types[i+1].get()))
            // TODO: should BasisType be included in here too?
            || (dynamic_cast<BitType *>(types[i].get())
                && dynamic_cast<BitType *>(types[i+1].get()))
            // TODO: Think about this recursively... technically
            //       bit[2][3] x bit[2] should not be canon.
            || ((left_broad = dynamic_cast<BroadcastType *>(types[i].get()))
                && (right_broad = dynamic_cast<BroadcastType *>(types[i+1].get()))
                && *left_broad->elem_type == *right_broad->elem_type)) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<Type> TupleType::canonicalize() const {
    // TODO: assumes that no type can be canonicalized into a QubitType or
    //       BitType. True today, maybe not tomorrow
    #define MERGE_ARR_TYPE(name, varname) \
        std::unique_ptr<DimVarExpr> dim = std::move(varname->dim->copy()); \
        size_t j; \
        const name##Type *next##name; \
        for (j = i+1; j < types.size() \
                && (next##name = dynamic_cast<const name##Type *>(types[j].get())); \
                j++) { \
            if (next##name) { \
                *dim += *next##name->dim; \
            } \
        } \
        canon_types.push_back(std::move(std::make_unique<name##Type>(std::move(dim)))); \
        i = j;

    std::vector<std::unique_ptr<Type>> canon_types;
    for (size_t i = 0; i < types.size(); ) {
        std::unique_ptr<Type> canon_type;
        const Type *this_type = types[i].get();
        if (!this_type->isCanonical()) {
            canon_type = std::move(this_type->canonicalize());
            this_type = canon_type.get();
        }

        if (const QubitType *qubitType = dynamic_cast<const QubitType *>(this_type)) {
            MERGE_ARR_TYPE(Qubit, qubitType)
        } else if (const BitType *bitType = dynamic_cast<const BitType *>(this_type)) {
            MERGE_ARR_TYPE(Bit, bitType)
        } else if (const BroadcastType *broad = dynamic_cast<const BroadcastType *>(this_type)) {
            std::unique_ptr<DimVarExpr> factor = broad->factor->copy();
            size_t j;
            const BroadcastType *next_broad;
            for (j = i+1; j < types.size()
                    && (next_broad = dynamic_cast<const BroadcastType *>(types[j].get()))
                    // TODO: bit[1] should be equal to bit[2] in theory (see isCanonical() comment)
                    && *next_broad->elem_type == *broad->elem_type;
                    j++) {
                *factor += *next_broad->factor;
            }
            canon_types.push_back(std::move(std::make_unique<BroadcastType>(
                broad->elem_type->copy(), std::move(factor))));
            i = j;
        } else {
            if (canon_type) {
                canon_types.push_back(std::move(canon_type));
            } else {
                canon_types.push_back(std::move(this_type->copy()));
            }
            i++;
        }
    }

    if (canon_types.size() == 1) {
        return std::move(canon_types[0]);
    } else {
        return std::move(std::make_unique<TupleType>(std::move(canon_types)));
    }
}

std::unique_ptr<Type> TupleType::unfurl() const {
    std::vector<std::unique_ptr<Type>> unfurled_types;
    for (size_t i = 0; i < types.size(); i++) {
        std::unique_ptr<Type> unfurled = types[i]->unfurl();
        // TODO: This needs to be fixed. Simple approach: instead of having an
        //       arg_names, a Kernel has args, each of which is an Argument
        //       with a name, start_idx, and end_idx. This allows Unit to be an
        //       argument, for example.
        assert(!dynamic_cast<TupleType *>(unfurled.get()) && "nested tuples are not allowed");
        unfurled_types.push_back(std::move(unfurled));
    }

    if (unfurled_types.size() == 1) {
        return std::move(unfurled_types[0]);
    } else {
        return std::move(std::make_unique<TupleType>(std::move(unfurled_types)));
    }
}

std::unique_ptr<Type> FuncType::asEmbedded(DebugInfo &dbg, EmbeddingKind kind) const {
    std::unique_ptr<Type> canon_type = canonicalize();
    FuncType *func_type;
    DimVarExpr *lhs_dim, *rhs_dim;

    if (!(func_type = dynamic_cast<FuncType *>(canon_type.get()))
            || !(lhs_dim = extractBitDim(func_type->lhs.get()))
            || !(rhs_dim = extractBitDim(func_type->rhs.get()))) {
        throw TypeException("Can only classically embed a variable as a "
                            "if it is a function from bits to bits.",
                            std::move(dbg.copy()));
    }

    std::unique_ptr<DimVarExpr> n_qubits;
    if (kind == EMBED_XOR) {
        // bit[M] -> bit[N] becomes qubit[M+N] -> qubit[M+N]
        // because XOR embedding preserves input
        n_qubits = lhs_dim->copy();
        *n_qubits += *rhs_dim;
    } else if (kind == EMBED_SIGN) {
        // bit[M] -> bit[N] becomes qubit[M] -> qubit[M]
        // because this looks like U|x> = (-1)^f(x)|x>
        n_qubits = lhs_dim->copy();
    } else if (kind == EMBED_INPLACE) {
        // No point in trying to do this comparison if DimVarExprs have not been
        // evaluated yet. This method (asEmbedded()) may get called from the
        // Python frontend before typechecking is even performed. So we'll
        // leave this validation to the type checker, who will call this method
        // again once the dimensions are constant.
        if (lhs_dim->isConstant()
                && rhs_dim->isConstant()
                && *lhs_dim != *rhs_dim) {
            throw TypeException("Cannot create an in-place embedding of a "
                                "classical function if its number of input "
                                "bits " + lhs_dim->toString() + " does "
                                "not match the number of output bits "
                                + rhs_dim->toString(),
                                std::move(dbg.copy()));
        }
        // U|x> = |f(x)>. We are trusting the programmer that the function is
        // truly reversible
        n_qubits = lhs_dim->copy();
    } else {
        throw TypeException("Compiler bug: Embedding kind "
                            + std::to_string(kind)
                            + " not handled",
                            std::move(dbg.copy()));
    }

    return std::make_unique<FuncType>(
            std::make_unique<QubitType>(n_qubits->copy()),
            std::make_unique<QubitType>(n_qubits->copy()),
            /*is_rev=*/true);
}

///////////////////////// DIMVAR INFERENCE /////////////////////////
std::unique_ptr<BroadcastType> Type::collapseToBroadcast(const BroadcastType &genericType) const {
    return std::make_unique<BroadcastType>(
        std::move(copy()),
        std::make_unique<DimVarExpr>("", 1));
}

std::unique_ptr<Type> BroadcastType::unfurl() const {
    assert(factor->isConstant() && "Unfurl will not work until dimvarexprs are evaluated");
    // TODO: very inefficient for Qubits and Bits. Need to optimize this for that case
    std::vector<std::unique_ptr<Type>> types;
    types.reserve(factor->offset);
    for (DimVarValue i = 0; i < factor->offset; i++) {
        types.push_back(std::move(elem_type->copy()));
    }
    std::unique_ptr<TupleType> tuple = std::make_unique<TupleType>(std::move(types));
    return std::move(tuple->canonicalize());
}

DimVarInferenceTypeMismatch::DimVarInferenceTypeMismatch(const Type &constType, const Type &genericType)
        : DimVarInferenceConflict("Type mismatch: " + constType.toString()
                                   + " is different from "
                                   + genericType.toString()) {}

void DimVarExpr::inferDimvars(const DimVarExpr &constDimVarExpr, DimVarValues &out) const {
    if (isConstant()) {
        return;
    } else if (dimvars.size() > 1) {
        // We don't have the code written (or the logic figured out) for
        // expressions with more than one dimvar. So just exit early.
        return;
    }

    DimVar dimvar = dimvars.begin()->first;
    // TODO: what about more complicated situations? I'm sure we can set
    //       some reasonable restrictions to make this similarly easy
    // Where x,y are positive integers and N is a dimvar:
    // x = z*N + y <-> N = (x-y)/z
    // Thus we need to check that x >= y to make sure N has a legit value
    assert(constDimVarExpr.isConstant() && "supposedly const DimVarExpr is not constant!");
    if (constDimVarExpr.offset < offset) {
        throw DimVarInferenceConflict("Type variable " + dimvar + " cannot be negative");
    }
    DimVarValue diff = constDimVarExpr.offset - offset;
    DimVarValue coeff = dimvars.at(dimvar);
    if (!coeff) {
        // TODO: handle this better
        throw DimVarInferenceConflict("Type variable " + dimvar + " has zero "
                                       "coefficient. I can't handle this, sorry");
    } else if (coeff < 0) {
        throw DimVarInferenceConflict("Type variable " + dimvar + " has "
                                       "negative coefficient "
                                       + std::to_string(coeff)
                                       + ". Cannot infer types.");
    } else if (diff % coeff) {
        throw DimVarInferenceConflict(
                "Type variable " + dimvar + " coefficient "
                + std::to_string(coeff) + " does not cleanly divide "
                + std::to_string(diff) + ". Cannot infer types.");
    }
    DimVarValue dimvarValue = diff / coeff;
    if (out.count(dimvar) && out[dimvar] != dimvarValue) {
        throw DimVarInferenceConflict("Conflicting values for " + dimvar
                                       + ": " + std::to_string(dimvarValue)
                                       + " != " + std::to_string(out[dimvar]));
    }
    out[dimvar] = dimvarValue;
}

void DimVarExpr::eval(DimVarValues &dimvar_values, bool permissive) {
    if (isConstant()) {
        if (offset < 0) {
            throw NegativeDimVarExprException();
        }
    } else {
        std::vector<DimVar> offending_dimvars;
        DimVarValue new_offset = offset;
        for (const std::pair<DimVar, DimVarValue> dv_val : dimvars) {
            if (!dimvar_values.count(dv_val.first)) {
                if (permissive) {
                    // Best effort. That's okay if we failed
                    return;
                } else {
                    throw MissingDimVarException(dv_val.first);
                }
            }
            new_offset += dv_val.second * dimvar_values[dv_val.first];
            offending_dimvars.push_back(dv_val.first);
        }
        offset = new_offset;
        dimvars.clear();

        if (offset < 0) {
            // TODO: pick some dimvar to throw here
            throw NegativeDimVarExprException(offending_dimvars);
        }
    }
}

void Kernel::typeCheck() {
    runASTVisitorPipeline();
    inferCalleeDimvars();
    findCaptureInstantiations();
}

void Kernel::findCaptureInstantiations() {
    // Walk the AST and collect all the instantiations
    {
        FindInstantiationsVisitor instantiator(*this);
        walk(instantiator);
    }

    for (size_t i = 0; i < capture_objs.size(); i++) {
        if (capture_objs[i]->needsExplicitDimvars() && capture_instances[i]->empty()) {
            std::vector<DimVar> missing_dimvars;
            capture_objs[i]->getMissingDimvars(missing_dimvars);
            std::ostringstream joined;
            bool first = true;
            for (DimVar dv : missing_dimvars) {
                if (!first) {
                    joined << ", ";
                }
                joined << dv;
                first = false;
            }

            throw TypeException("Capture " + capture_names[i] + " still needs "
                                "dimvars (" + joined.str() + ") and hasn't "
                                "been instantiated",
                                std::move(dbg->copy()));
        }
    }
}

void Kernel::inferCalleeDimvars() {
    // Now, at this point, capture_types contains constant types (i.e., types
    // with no dimvars). So now we can use those to infer the dimvars for
    // captures... hopefully
    assert(capture_objs.size() == capture_types.size()
           && "Number of capture objects does not match number of capture types");
    for (size_t i = 0; i < capture_objs.size(); i++) {
        HybridObj &obj = *capture_objs[i];
        size_t freevars = capture_freevars[i];
        if (!obj.getType().isConstant()) {
            DimVarValues vals;
            try {
                obj.getType().inferDimvars(*capture_types[i], vals);
                if (vals.empty()) {
                    throw DimVarInferenceConflict("callee inference failed");
                }
            } catch (DimVarInferenceConflict &exc) {
                throw TypeException("Cannot infer type variables for callee. "
                                    + exc.message + " at capture #"
                                    + std::to_string(i+1) + " ("
                                    + capture_names[i] + ") of "
                                    + ast_kind_name(kind()) + " decorator for "
                                    "function " + name + "()",
                                    std::move(dbg->copy()));
            }
            std::unique_ptr<HybridObj> copy = obj.copyHybrid();
            copy->evalExplicitDimvars(vals, freevars);
            capture_objs[i] = std::move(copy);
        }

        if (!(capture_objs[i]->getType() <= *capture_types[i])) {
            throw TypeException("Wrong type for capture #"
                                + std::to_string(i+1) + " ("
                                + capture_names[i] + ") of function "
                                + name + "(). Expected "
                                + capture_types[i]->toString() + " but got "
                                + capture_objs[i]->getType().toString(),
                                std::move(dbg->copy()));
        }
    }
}

void Kernel::evalExplicitDimvars(DimVarValues &new_vals, size_t n_freevars) {
    assert(!new_vals.empty()
           && "evalExplicitDimvars() called with no dimvar values");

    std::vector<std::optional<DimVarValue>> values;
    values.reserve(new_vals.size());
    for (DimVar dimvar : dimvars) {
        if (new_vals.count(dimvar)) {
            assert(!dimvar_values.count(dimvar)
                   && "Dimvar inferred by both capture-based dimvar "
                      "inference and caller-based dimvar inference!");
            values.push_back(new_vals[dimvar]);
        } else if (!dimvar_values.count(dimvar)) {
            if (n_freevars) {
                n_freevars--;
                values.push_back(std::nullopt);
            } else {
                throw TypeException("You need to specify more free variables "
                                    "to kernel " + name
                                    + "() in the capture where it is captured "
                                    "by another kernel", std::move(dbg->copy()));
            }
        }
    }

    registerExplicitDimvars(values);
    assert(dimvar_values.size() == dimvars.size()
           && "Not all dimvars have values???");
    typeCheck();

    // Imitates the behavior of jit.py
    if (qwerty_debug) {
        std::string py_ext = ".py";
        std::string basename;
        if (dbg->srcfile.size() >= py_ext.size()
                && !dbg->srcfile.compare(dbg->srcfile.size()-py_ext.size(),
                                         py_ext.size(), py_ext)) {
            size_t idx = dbg->srcfile.rfind('.');
            basename = dbg->srcfile.substr(0, idx);
        } else {
            basename = dbg->srcfile;
        }
        std::string filename = basename + "_" + name + ".dot";

        std::ofstream stream(filename);
        GraphvizVisitor visitor;
        walk(visitor);
        stream << visitor.str();
        stream.close();
    }
}

void ClassicalKernel::typeCheck() {
    FuncType &funcType = dynamic_cast<FuncType &>(*type);
    if (!funcType.lhs->isReversibleFriendly() || !funcType.rhs->isReversibleFriendly()) {
        // TODO: better error message
        throw TypeException("Function type " + type->toString() + " is not "
                            "purely classical as required for functions "
                            "decorated with " + ast_kind_name(kind()),
                            std::move(dbg->copy()));
    }

    for (size_t i = 0; i < capture_types.size(); i++) {
        if (!capture_types[i]->isReversibleFriendly()) {
            throw TypeException("Capture #" + std::to_string(i+1) + " ("
                                + capture_names[i] + ") of " + name + "() is "
                                "not purely classical as required for "
                                "functions decorated with "
                                + ast_kind_name(kind()),
                                std::move(dbg->copy()));
        }
    }

    Kernel::typeCheck();

    [[maybe_unused]] FuncType &funcTypeAfter = dynamic_cast<FuncType &>(*type);
    assert(funcTypeAfter.lhs->isReversibleFriendly()
           && funcTypeAfter.rhs->isReversibleFriendly()
           && "Somehow after type checking, @classical type became non-classical!");
}

void Kernel::inferDimvarsFromCaptures(std::vector<std::unique_ptr<HybridObj>> provided_capture_objs) {
    assert(dimvar_values.empty() && "Calling inferDimvars() twice?");

    capture_objs = std::move(provided_capture_objs);
    for (size_t i = 0; i < capture_objs.size(); i++) {
        // Fill with empty maps
        capture_instances.push_back(std::make_unique<std::map<const std::vector<DimVarValue>, std::unique_ptr<HybridObj>>>());
    }

    if (capture_objs.size() != capture_types.size()) {
        throw TypeException("Wrong number of captures passed to "
                            + ast_kind_name(kind()) + " decorator for "
                            "function " + name + "(). Expected "
                            + std::to_string(capture_types.size()) + " but got "
                            + std::to_string(capture_objs.size()),
                            std::move(dbg->copy()));
    }

    for (size_t i = 0; i < capture_objs.size(); i++) {
        const Type &provided_type = capture_objs[i]->getType();

        if (!provided_type.isConstant()) {
            // This is a kernel we captured that itself needs explicit dimvars
            // if it were called on its own. Hopefully, we can infer the
            // dimvars it needs to make this whole situation well-typed.
        } else {
            try {
                capture_types[i]->inferDimvars(provided_type, dimvar_values);
            } catch (DimVarInferenceConflict &exc) {
                throw TypeException("Cannot infer type variables. " + exc.message
                                    + " at capture #" + std::to_string(i+1) + " ("
                                    + capture_names[i] + ") of "
                                    + ast_kind_name(kind()) + " decorator for "
                                    "function " + name + "()",
                                    std::move(dbg->copy()));
            }
        }
    }

    missing_dimvars.clear();
    needs_explicit_dimvars = false;
    for (DimVar dimvar : dimvars) {
        if (!dimvar_values.count(dimvar)) {
            needs_explicit_dimvars = true;
            missing_dimvars.push_back(dimvar);
        }
    }
    type->evalDimVarExprs(dimvar_values, true);
    if (type->isConstant() && type->isFurled()) {
        type = std::move(type->unfurl());
    }
}

void Kernel::registerExplicitDimvars(std::vector<std::optional<DimVarValue>> &values) {
    std::vector<DimVar> fresh_dimvars;

    for (DimVar dimvar : dimvars) {
        if (!dimvar_values.count(dimvar)) {
            fresh_dimvars.push_back(dimvar);
        }
    }

    // Used in obscure situations (namely invoking a @classical kernel
    // classically without any Qwerty JITing)
    this->explicit_dimvars = fresh_dimvars;

    if (values.size() != fresh_dimvars.size()) {
        std::ostringstream list;
        for (size_t i = 0; i < fresh_dimvars.size(); i++) {
            if (i) {
                list << ", ";
            }
            list << fresh_dimvars[i];
        }

        throw TypeException("Wrong number of explicit dimvar values passed. "
                            "Expected " + std::to_string(fresh_dimvars.size())
                            + (list.str().empty()? "" : " (for " + list.str() + ")")
                            + " but got " + std::to_string(values.size()),
                            std::move(dbg->copy()));
    }

    missing_dimvars.clear();
    needs_explicit_dimvars = false;

    for (size_t i = 0; i < fresh_dimvars.size(); i++) {
        if (values[i]) {
            dimvar_values[fresh_dimvars[i]] = values[i].value();
        } else {
            needs_explicit_dimvars = true;
            missing_dimvars.push_back(fresh_dimvars[i]);
        }
    }

    type->evalDimVarExprs(dimvar_values, true);
    if (type->isConstant() && type->isFurled()) {
        type = std::move(type->unfurl());
    }
}
