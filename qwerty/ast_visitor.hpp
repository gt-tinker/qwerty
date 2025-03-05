#ifndef AST_VISITOR_H
#define AST_VISITOR_H

#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mockturtle/networks/xag.hpp"
#include "defs.hpp"

// The kind of traversal used in visiting AST nodes
enum class Traversal {
    PREORDER,
    POSTORDER,
    // This is both preorder and postorder
    PREPOSTORDER,
    // The AST visitor driver will only call visit() for the root node, and it
    // is the responsibility of the visitor itself to recursively call visit()
    // on child nodes. Useful when you have a very specific order in which you
    // want to traverse the AST.
    CUSTOM
};

// Holds additional information about the AST traversal, explained
// field-by-field below.
struct ASTVisitContext {
    // A reference to the parent node. If this is the root node, this is a
    // reference to the node itself instead.
    ASTNode &parent;
    // The label returned along with this child when ASTNode::children() was
    // called on the parent.
    std::string label;
    // A reference to the field of the parent that points to this child. Very
    // useful for replacing a node in-place by changing its parent's pointer to
    // point to a new/different node.
    std::unique_ptr<ASTNode> &ptr;

    ASTVisitContext(ASTNode &parent,
                    std::string label,
                    std::unique_ptr<ASTNode> &ptr)
                   : parent(parent), label(label), ptr(ptr) {}
};

// The abstract base class for an AST visitor. There must be a visit() method
// for every node (search for VISITOR_TEDIUM in ast.hpp to see how this is
// accomplished in C++). A visit() method returns false to alert the AST visitor
// driver code that the node visited has been replaced in-place that its
// replacement should be visited now instead of continuing with the traversal.
// Otherwise, visit() returns true (most cases).
struct ASTVisitor {
    virtual ~ASTVisitor() {}
    virtual Traversal traversal() = 0;
    virtual void init(ASTNode &root) {}
    virtual void finish() {}

    virtual bool visit(ASTVisitContext &ctx, Assign &assign) = 0;
    virtual bool visit(ASTVisitContext &ctx, DestructAssign &dassign) = 0;
    virtual bool visit(ASTVisitContext &ctx, Return &ret) = 0;
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) = 0;
    virtual bool visit(ASTVisitContext &ctx, Variable &var) = 0;
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) = 0;
    // @qpu nodes
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) = 0;
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) = 0;
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) = 0;
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) = 0;
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) = 0;
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) = 0;
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) = 0;
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) = 0;
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) = 0;
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) = 0;
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) = 0;
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) = 0;
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) = 0;
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) = 0;
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) = 0;
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) = 0;
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) = 0;
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) = 0;
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) = 0;
    virtual bool visit(ASTVisitContext &ctx, Identity &id) = 0;
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &id) = 0;
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) = 0;
    virtual bool visit(ASTVisitContext &ctx, Measure &meas) = 0;
    virtual bool visit(ASTVisitContext &ctx, Project &proj) = 0;
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) = 0;
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) = 0;
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) = 0;
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) = 0;
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) = 0;
    // @classical nodes
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) = 0;
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) = 0;
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) = 0;
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) = 0;
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) = 0;
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) = 0;
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) = 0;
};

// Some passes like the graphviz generator don't care what specific nodes they
// are visiting. This is helpful for that case, instead defining a single
// visitNode() method that is called for every node.
struct ObliviousASTVisitor : ASTVisitor {
    virtual bool visitNode(ASTVisitContext &ctx, ASTNode &node) = 0;

    virtual bool visit(ASTVisitContext &ctx, Assign &assign) override { return visitNode(ctx, (ASTNode &)assign); }
    virtual bool visit(ASTVisitContext &ctx, DestructAssign &dassign) override { return visitNode(ctx, (ASTNode &)dassign); }
    virtual bool visit(ASTVisitContext &ctx, Return &ret) override { return visitNode(ctx, (ASTNode &)ret); }
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) override { return visitNode(ctx, (ASTNode &)kernel); }
    virtual bool visit(ASTVisitContext &ctx, Variable &var) override { return visitNode(ctx, (ASTNode &)var); }
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override { return visitNode(ctx, (ASTNode &)slice); }
    // @qpu nodes
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) override { return visitNode(ctx, (ASTNode &)adj); }
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) override { return visitNode(ctx, (ASTNode &)prep); }
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) override { return visitNode(ctx, (ASTNode &)lift); }
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) override { return visitNode(ctx, (ASTNode &)embed); }
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override { return visitNode(ctx, (ASTNode &)pipe); }
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override { return visitNode(ctx, (ASTNode &)inst); }
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override { return visitNode(ctx, (ASTNode &)repeat); }
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override { return visitNode(ctx, (ASTNode &)reptens); }
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) override { return visitNode(ctx, (ASTNode &)pred); }
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override { return visitNode(ctx, (ASTNode &)broadtensor); }
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) override { return visitNode(ctx, (ASTNode &)bitensor); }
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override { return visitNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) override { return visitNode(ctx, (ASTNode &)phase); }
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) override { return visitNode(ctx, (ASTNode &)float_); }
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) override { return visitNode(ctx, (ASTNode &)neg); }
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) override { return visitNode(ctx, (ASTNode &)bin); }
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override { return visitNode(ctx, (ASTNode &)fdve); }
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) override { return visitNode(ctx, (ASTNode &)tuple); }
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override { return visitNode(ctx, (ASTNode &)std); }
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override { return visitNode(ctx, (ASTNode &)id); }
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &trans) override { return visitNode(ctx, (ASTNode &)trans); }
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override { return visitNode(ctx, (ASTNode &)discard); }
    virtual bool visit(ASTVisitContext &ctx, Measure &meas) override { return visitNode(ctx, (ASTNode &)meas); }
    virtual bool visit(ASTVisitContext &ctx, Project &proj) override { return visitNode(ctx, (ASTNode &)proj); }
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) override { return visitNode(ctx, (ASTNode &)flip); }
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) override { return visitNode(ctx, (ASTNode &)rot); }
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) override { return visitNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) override { return visitNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) override { return visitNode(ctx, (ASTNode &)cond); }
    // @classical nodes
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) override { return visitNode(ctx, (ASTNode &)unOp); }
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) override { return visitNode(ctx, (ASTNode &)binOp); }
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) override { return visitNode(ctx, (ASTNode &)reduceOp); }
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) override { return visitNode(ctx, (ASTNode &)concat); }
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) override { return visitNode(ctx, (ASTNode &)repeat); }
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) override { return visitNode(ctx, (ASTNode &)mulOp); }
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override { return visitNode(ctx, (ASTNode &)bitLit); }
};

// The job of this guy is to verify that every expression of an immaterial type
// (e.g,. Basis[N]) is used only as operands to AST nodes like basis
// translations, not e.g., the right-hand side of an assignment. In the cases
// of bases, this is because ultimately a basis in Qwerty is part of the syntax
// of basis-oriented constructs, not an actual expression.
struct FlagImmaterialVisitor : ObliviousASTVisitor {
    virtual Traversal traversal() override { return Traversal::POSTORDER; } // Doesn't matter
    virtual void init(ASTNode &root) override {}

    virtual bool visitNode(ASTVisitContext &ctx, ASTNode &node) override;
};

// Shared type checking code for @classical and @qpu kernels
struct BaseTypeCheckVisitor : ASTVisitor {
    bool is_rev;
    std::unordered_set<DimVar> dimvars;
    std::unordered_map<std::string, std::unique_ptr<Type>> variables;
    std::unordered_set<std::string> linear_vars_used;

    BaseTypeCheckVisitor() : is_rev(true) {}

    virtual Traversal traversal() override { return Traversal::CUSTOM; }
    virtual void init(ASTNode &root) override;
    void validateSlice(Slice &slice, DimVarValue dim, DimVarValue &n_elem,
                       DimVarValue &start_idx);

    virtual bool visit(ASTVisitContext &ctx, Assign &assign) override;
    virtual bool visit(ASTVisitContext &ctx, DestructAssign &dassign) override;
    virtual bool visit(ASTVisitContext &ctx, Return &ret) override;
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) override;
    virtual bool visit(ASTVisitContext &ctx, Variable &var) override;
};

// Type checking specifically for @qpu kernels
struct QpuTypeCheckVisitor : BaseTypeCheckVisitor {
    // The primitive basis for a range of qubits
    struct PrimRange {
        PrimitiveBasis prim_basis;
        size_t start, end; // [start, end)

        PrimRange(PrimitiveBasis prim_basis,
                  size_t start,
                  size_t end)
                 : prim_basis(prim_basis),
                   start(start),
                   end(end) {}

        bool operator==(const PrimRange &pd) const {
            return prim_basis == pd.prim_basis
                   && start == pd.start
                   && end == pd.end;
        }

        bool operator!=(const PrimRange &pd) const {
            return !(*this == pd);
        }
    };

    // Interpret individual basis vectors as singleton basis literals
    std::unique_ptr<ASTNode> wrapBasis(ASTVisitContext &ctx,
                                       std::unique_ptr<ASTNode> node,
                                       bool allow_empty);
    bool onlyLiterals(ASTNode *node);
    bool singletonBasis(ASTNode *node);
    PrimitiveBasis basisVectorPrimitiveBasisOrError(ASTNode *node, size_t vector_idx);
    void extractPrimRanges(ASTNode &node, std::vector<PrimRange> &prim_ranges_out);
    llvm::APInt basisVectorEigenbitsOrError(ASTNode *node, size_t vector_idx);
    bool isQfunc(const Type *type, DebugInfo &dbg, bool must_be_rev, DimVarExpr **dim_out);
    std::unique_ptr<Type> biTensorTypes(const Type *left, const Type *right);
    std::unique_ptr<Type> broadcastTensorType(const Type *type, const DimVarExpr &factor);

    // Don't hide visit()s in base class
    using BaseTypeCheckVisitor::visit;

    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override;
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) override;
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) override;
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) override;
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) override;
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override;
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override;
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override;
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override;
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) override;
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) override;
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override;
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override;
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) override;
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) override;
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) override;
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) override;
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override;
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) override;
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override;
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override;
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &trans) override;
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override;
    virtual bool visit(ASTVisitContext &ctx, Measure &measure) override;
    virtual bool visit(ASTVisitContext &ctx, Project &proj) override;
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) override;
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) override;
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) override;
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) override;
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) override;
    // @classical nodes
    virtual bool visitNonQpuNode(ASTVisitContext &ctx, ASTNode &node);
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) override { return visitNonQpuNode(ctx, (ASTNode &)unOp); }
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) override { return visitNonQpuNode(ctx, (ASTNode &)binOp); }
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) override { return visitNonQpuNode(ctx, (ASTNode &)reduceOp); }
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) override { return visitNonQpuNode(ctx, (ASTNode &)concat); }
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) override { return visitNonQpuNode(ctx, (ASTNode &)repeat); }
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) override { return visitNonQpuNode(ctx, (ASTNode &)mulOp); }
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override { return visitNonQpuNode(ctx, (ASTNode &)bitLit); }
};

// Type checking specifically for @classical kernels
struct ClassicalTypeCheckVisitor : BaseTypeCheckVisitor {
    // Don't hide visit()s in base class
    using BaseTypeCheckVisitor::visit;

    // @qpu nodes
    virtual bool visitNonClassicalNode(ASTVisitContext &ctx, ASTNode &node);
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) override { return visitNonClassicalNode(ctx, (ASTNode &)adj); }
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) override { return visitNonClassicalNode(ctx, (ASTNode &)prep); }
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) override { return visitNonClassicalNode(ctx, (ASTNode &)lift); }
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) override { return visitNonClassicalNode(ctx, (ASTNode &)embed); }
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override { return visitNonClassicalNode(ctx, (ASTNode &)pipe); }
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override { return visitNonClassicalNode(ctx, (ASTNode &)inst); }
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override { return visitNonClassicalNode(ctx, (ASTNode &)repeat); }
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override { return visitNonClassicalNode(ctx, (ASTNode &)reptens); }
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) override { return visitNonClassicalNode(ctx, (ASTNode &)pred); }
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override { return visitNonClassicalNode(ctx, (ASTNode &)broadtensor); }
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) override { return visitNonClassicalNode(ctx, (ASTNode &)bitensor); }
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override { return visitNonClassicalNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) override { return visitNonClassicalNode(ctx, (ASTNode &)phase); }
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) override { return visitNonClassicalNode(ctx, (ASTNode &)float_); }
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) override { return visitNonClassicalNode(ctx, (ASTNode &)neg); }
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) override { return visitNonClassicalNode(ctx, (ASTNode &)bin); }
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override { return visitNonClassicalNode(ctx, (ASTNode &)fdve); }
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) override { return visitNonClassicalNode(ctx, (ASTNode &)tuple); }
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override { return visitNonClassicalNode(ctx, (ASTNode &)std); }
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override { return visitNonClassicalNode(ctx, (ASTNode &)id); }
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &trans) override { return visitNonClassicalNode(ctx, (ASTNode &)trans); }
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override { return visitNonClassicalNode(ctx, (ASTNode &)discard); }
    virtual bool visit(ASTVisitContext &ctx, Measure &meas) override { return visitNonClassicalNode(ctx, (ASTNode &)meas); }
    virtual bool visit(ASTVisitContext &ctx, Project &proj) override { return visitNonClassicalNode(ctx, (ASTNode &)proj); }
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) override { return visitNonClassicalNode(ctx, (ASTNode &)flip); }
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) override { return visitNonClassicalNode(ctx, (ASTNode &)rot); }
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) override { return visitNonClassicalNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) override { return visitNonClassicalNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) override { return visitNonClassicalNode(ctx, (ASTNode &)cond); }

    // @classical nodes
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) override;
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) override;
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override;
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override;
};

// Rewrites AST w/o syntax sugar. While technically a form of canonicalization,
// this visit pass is made distinct to ensure only a very small set of transformations
// can avoid type-checking.
struct DesugarVisitor : ObliviousASTVisitor {
    virtual Traversal traversal() override { return Traversal::PREPOSTORDER; }
    virtual void init(ASTNode &root) override {}

    virtual bool visitNode(ASTVisitContext &ctx, ASTNode &node) override;
};

// Simplifies the AST. Performs some basic optimizations (Section 4.2 of the
// CGO paper) and also replaces some semi-redundant nodes. For example,
// BroadcastTensors and LiftBits do not survive this visitor.
struct CanonicalizeVisitor : ASTVisitor {
    virtual Traversal traversal() override { return Traversal::PREPOSTORDER; }
    virtual void init(ASTNode &root) override {}

    virtual bool visit(ASTVisitContext &ctx, Assign &assign) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, DestructAssign &assign) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Return &ret) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Variable &var) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override { return true; }
    // @qpu nodes
    // This actually does something!
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) override;
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) override { return true; }
    // This actually does something!
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) override;
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override { return true; }
    // This actually does something!
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) override;
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) override { return true; }
    // This actually does something!
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override;
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) override { return true; }
    // This actually does something!
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) override;
    // This actually does something!
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) override;
    // This should have been removed by EvalDimVarExprVisitor
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &trans) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Measure &measure) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Project &proj) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) override { return true; }
    // @classical nodes
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) override { return true; }
    // TODO: merge concatenated bit literals
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) override { return true; }
    // TODO: merge repeated bit literals
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &concat) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override { return true; }
};

// Evaluate all dimension variables, leaving all dimension variables in the
// whole AST constant.
struct EvalDimVarExprVisitor : ObliviousASTVisitor {
    DimVarValues &dimvar_values;

    EvalDimVarExprVisitor(DimVarValues &dimvar_values)
                        : dimvar_values(dimvar_values) {}

    virtual Traversal traversal() override { return Traversal::PREORDER; }
    virtual void init(ASTNode &root) override {}

    // Don't hide visit()s in base class
    using ObliviousASTVisitor::visit;

    virtual bool visitNode(ASTVisitContext &ctx, ASTNode &node) override;
    // Has DimVarExprs permeating it
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) override;
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override;
    // @qpu nodes
    // Need to expand this to a chain of Pipe nodes
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override;
    // Right-hand side is a DimVarExpr
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override;
    // Throw an error if we see this — the Pipe visitor above should've gotten rid of these!
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override;
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override;
    // Has DimVarExpr factor. And if it's zero, we should replace this with a UnitLiteral
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override;
    // Has QubitType, which contains DimVarExpr dim
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override;
    // Has BasisType, which contains DimVarExpr dim
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override;
    // Has FuncType, which contains QubitTypes with DimVarExpr dims
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override;
    // Has FuncType, which contains QubitTypes with DimVarExpr dims
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override;
    // Has a DimVarExpr to be used as a constant in a float expression
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override;
    // @classical nodes
    // Has a DimVarExpr saying how many times to repeat the bit
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) override;
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mul) override;
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override;
};

// Used by Kernel::findCaptureInstantiations() to find all Instantiate nodes
// (e.g., my_capture[[35]]) in a kernel AST
struct FindInstantiationsVisitor : ObliviousASTVisitor {
    Kernel &kernel;

    FindInstantiationsVisitor(Kernel &kernel)
                   : kernel(kernel) {}

    void materializeInstantiation(
        std::string capture_name,
        const std::vector<DimVarValue> &instantiate_val,
        DebugInfo &dbg);

    virtual Traversal traversal() override { return Traversal::PREORDER; }
    virtual void init(ASTNode &root) override {}

    // Don't hide visit()s in base class
    using ObliviousASTVisitor::visit;

    virtual bool visitNode(ASTVisitContext &ctx, ASTNode &node) override { return true; }
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override;
};

// Generate a .dot file from a Qwerty AST. Very useful for debugging. (See
// docs/debugging.md.)
struct GraphvizVisitor : ObliviousASTVisitor {
    bool print_dbg;
    std::unordered_map<ASTNode *, size_t> node_indices;
    size_t next_node_index;
    std::ostringstream ss;

    GraphvizVisitor() : print_dbg(false) {}
    GraphvizVisitor(bool print_dbg) : print_dbg(print_dbg) {}

    virtual Traversal traversal() override { return Traversal::PREORDER; }
    virtual bool visitNode(ASTVisitContext &ctx, ASTNode &node) override;
    virtual void init(ASTNode &root) override;
    virtual void finish() override;

    std::string str();
    size_t getNodeIndex(ASTNode *node);
    void drawEdge(ASTNode *from, ASTNode *to, std::string edge_label);
};

// Generate a mockturtle netlist from a typechecked @classical AST
struct ClassicalNetlistVisitor : ASTVisitor {
    std::vector<std::unique_ptr<HybridObj>> &provided_captures;
    mockturtle::xag_network net;
    std::unordered_map<std::string, std::vector<mockturtle::xag_network::signal>> variable_wires;
    std::vector<uint32_t> output_wires;

    ClassicalNetlistVisitor(
            std::vector<std::unique_ptr<HybridObj>> &provided_captures)
            : provided_captures(provided_captures) {}

    virtual Traversal traversal() override { return Traversal::POSTORDER; }
    virtual void init(ASTNode &root) override;

    // Shared nodes for @qpu and @classical
    virtual bool visit(ASTVisitContext &ctx, Assign &assign) override;
    virtual bool visit(ASTVisitContext &ctx, DestructAssign &assign) override;
    virtual bool visit(ASTVisitContext &ctx, Return &ret) override;
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) override;
    virtual bool visit(ASTVisitContext &ctx, Variable &var) override;
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override;

    // @qpu nodes
    virtual bool visitNonClassicalNode(ASTVisitContext &ctx, ASTNode &node);
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) override { return visitNonClassicalNode(ctx, (ASTNode &)adj); }
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) override { return visitNonClassicalNode(ctx, (ASTNode &)prep); }
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) override { return visitNonClassicalNode(ctx, (ASTNode &)lift); }
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) override { return visitNonClassicalNode(ctx, (ASTNode &)embed); }
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override { return visitNonClassicalNode(ctx, (ASTNode &)pipe); }
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override { return visitNonClassicalNode(ctx, (ASTNode &)inst); }
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override { return visitNonClassicalNode(ctx, (ASTNode &)repeat); }
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override { return visitNonClassicalNode(ctx, (ASTNode &)reptens); }
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) override { return visitNonClassicalNode(ctx, (ASTNode &)pred); }
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override { return visitNonClassicalNode(ctx, (ASTNode &)broadtensor); }
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) override { return visitNonClassicalNode(ctx, (ASTNode &)bitensor); }
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override { return visitNonClassicalNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) override { return visitNonClassicalNode(ctx, (ASTNode &)phase); }
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) override { return visitNonClassicalNode(ctx, (ASTNode &)float_); }
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) override { return visitNonClassicalNode(ctx, (ASTNode &)neg); }
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) override { return visitNonClassicalNode(ctx, (ASTNode &)bin); }
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override { return visitNonClassicalNode(ctx, (ASTNode &)fdve); }
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) override { return visitNonClassicalNode(ctx, (ASTNode &)tuple); }
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override { return visitNonClassicalNode(ctx, (ASTNode &)std); }
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override { return visitNonClassicalNode(ctx, (ASTNode &)id); }
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &trans) override { return visitNonClassicalNode(ctx, (ASTNode &)trans); }
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override { return visitNonClassicalNode(ctx, (ASTNode &)discard); }
    virtual bool visit(ASTVisitContext &ctx, Measure &meas) override { return visitNonClassicalNode(ctx, (ASTNode &)meas); }
    virtual bool visit(ASTVisitContext &ctx, Project &proj) override { return visitNonClassicalNode(ctx, (ASTNode &)proj); }
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) override { return visitNonClassicalNode(ctx, (ASTNode &)flip); }
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) override { return visitNonClassicalNode(ctx, (ASTNode &)rot); }
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) override { return visitNonClassicalNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) override { return visitNonClassicalNode(ctx, (ASTNode &)lit); }
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) override { return visitNonClassicalNode(ctx, (ASTNode &)cond); }

    // @classical nodes
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) override;
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) override;
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) override;
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override;
};

// Generate Qwerty-dialect MLIR from a typechecked @qpu AST
struct QpuLoweringVisitor : ASTVisitor {
    MlirHandle &handle;
    std::string funcOp_name;
    std::string temp_func_prefix;
    size_t n_temp_funcs;
    std::vector<std::unique_ptr<HybridObj>> &provided_captures;
    std::unordered_map<std::string, llvm::SmallVector<mlir::Value>> variable_values;
    // Memory is allocated in provided_captures above.
    // The vector of DimVarValues is empty for non-instantiates
    std::map<std::pair<std::string, const std::vector<DimVarValue>>, ClassicalKernel *> cfunc_names;
    std::map<std::pair<std::string, const std::vector<DimVarValue>>, llvm::SmallVector<mlir::Value>> instantiate_values;
    // An array of amplitudes cannot be lowered to MLIR (it is "immaterial," we
    // say). So instead store them here so we can lower this directly to a
    // SuperposOp
    std::unordered_map<std::string, std::vector<std::complex<double>>> amplitude_names;

    QpuLoweringVisitor(MlirHandle &handle,
                       std::string funcOp_name,
                       std::vector<std::unique_ptr<HybridObj>> &provided_captures)
                      : handle(handle),
                        funcOp_name(funcOp_name),
                        provided_captures(provided_captures) {}

    mlir::ValueRange wrapStationary(mlir::Location loc,
                                    mlir::TypeRange result_types,
                                    mlir::ValueRange args,
                                    std::function<void(mlir::ValueRange)> build_body);
    mlir::Value createLambda(mlir::Location loc,
                             const FuncType &func_type,
                             mlir::ValueRange captures,
                             std::function<void(mlir::ValueRange, mlir::ValueRange)> add_contents);
    void walkBasisList(ASTNode *node,
                       ASTVisitContext &ctx,
                       bool already_visited,
                       qwerty::PrimitiveBasis &prim_basis_out,
                       mlir::Value &theta_out,
                       llvm::APInt &eigenbits_out);
    void walkSuperposOperandHelper(
        ASTNode &node,
        qwerty::PrimitiveBasis &prim_basis_out,
        double &theta_out,
        llvm::APInt &eigenbits_out,
        llvm::SmallVector<qwerty::BasisVectorAttr> &vecs_out);
    void walkSuperposOperand(
        ASTNode &node,
        double &theta_out,
        llvm::SmallVector<qwerty::BasisVectorAttr> &vec_attrs);
    mlir::Value materializeSimpleCapture(DebugInfo &dbg,
                                         mlir::Location loc,
                                         HybridObj *capture);

    virtual Traversal traversal() override { return Traversal::CUSTOM; }
    virtual void init(ASTNode &root) override;

    // Shared nodes for @qpu and @classical
    virtual bool visit(ASTVisitContext &ctx, Assign &assign) override;
    virtual bool visit(ASTVisitContext &ctx, DestructAssign &dassign) override;
    virtual bool visit(ASTVisitContext &ctx, Return &ret) override;
    virtual bool visit(ASTVisitContext &ctx, Kernel &kernel) override;
    virtual bool visit(ASTVisitContext &ctx, Variable &var) override;
    virtual bool visit(ASTVisitContext &ctx, Slice &slice) override;
    // @qpu nodes
    virtual bool visit(ASTVisitContext &ctx, Adjoint &adj) override;
    virtual bool visit(ASTVisitContext &ctx, Prepare &prep) override;
    virtual bool visit(ASTVisitContext &ctx, Lift &lift) override;
    virtual bool visit(ASTVisitContext &ctx, EmbedClassical &embed) override;
    virtual bool visit(ASTVisitContext &ctx, Pipe &pipe) override;
    virtual bool visit(ASTVisitContext &ctx, Instantiate &inst) override;
    virtual bool visit(ASTVisitContext &ctx, Repeat &repeat) override;
    virtual bool visit(ASTVisitContext &ctx, RepeatTensor &reptens) override;
    virtual bool visit(ASTVisitContext &ctx, Pred &pred) override;
    virtual bool visit(ASTVisitContext &ctx, BiTensor &bitensor) override;
    virtual bool visit(ASTVisitContext &ctx, BroadcastTensor &broadtensor) override;
    virtual bool visit(ASTVisitContext &ctx, QubitLiteral &lit) override;
    virtual bool visit(ASTVisitContext &ctx, Phase &phase) override;
    virtual bool visit(ASTVisitContext &ctx, FloatLiteral &float_) override;
    virtual bool visit(ASTVisitContext &ctx, FloatNeg &neg) override;
    virtual bool visit(ASTVisitContext &ctx, FloatBinaryOp &bin) override;
    virtual bool visit(ASTVisitContext &ctx, FloatDimVarExpr &fdve) override;
    virtual bool visit(ASTVisitContext &ctx, TupleLiteral &tuple) override;
    virtual bool visit(ASTVisitContext &ctx, BuiltinBasis &std) override;
    virtual bool visit(ASTVisitContext &ctx, Identity &id) override;
    virtual bool visit(ASTVisitContext &ctx, BasisTranslation &trans) override;
    virtual bool visit(ASTVisitContext &ctx, Discard &discard) override;
    virtual bool visit(ASTVisitContext &ctx, Measure &meas) override;
    virtual bool visit(ASTVisitContext &ctx, Project &proj) override;
    virtual bool visit(ASTVisitContext &ctx, Flip &flip) override;
    virtual bool visit(ASTVisitContext &ctx, Rotate &rot) override;
    virtual bool visit(ASTVisitContext &ctx, BasisLiteral &lit) override;
    virtual bool visit(ASTVisitContext &ctx, SuperposLiteral &lit) override;
    virtual bool visit(ASTVisitContext &ctx, Conditional &cond) override;
    // @classical nodes
    virtual bool visitNonQpuNode(ASTVisitContext &ctx, ASTNode &node);
    virtual bool visit(ASTVisitContext &ctx, BitUnaryOp &unOp) override { return visitNonQpuNode(ctx, (ASTNode &)unOp); }
    virtual bool visit(ASTVisitContext &ctx, BitBinaryOp &binOp) override { return visitNonQpuNode(ctx, (ASTNode &)binOp); }
    virtual bool visit(ASTVisitContext &ctx, BitReduceOp &reduceOp) override { return visitNonQpuNode(ctx, (ASTNode &)reduceOp); }
    virtual bool visit(ASTVisitContext &ctx, BitConcat &concat) override { return visitNonQpuNode(ctx, (ASTNode &)concat); }
    virtual bool visit(ASTVisitContext &ctx, BitRepeat &repeat) override { return visitNonQpuNode(ctx, (ASTNode &)repeat); }
    virtual bool visit(ASTVisitContext &ctx, ModMulOp &mulOp) override { return visitNonQpuNode(ctx, (ASTNode &)mulOp); }
    virtual bool visit(ASTVisitContext &ctx, BitLiteral &bitLit) override { return visitNonQpuNode(ctx, (ASTNode &)bitLit); }
};

#endif
