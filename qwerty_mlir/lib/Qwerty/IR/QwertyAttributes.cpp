//===- QwertyAttributes.cpp - Qwerty dialect attributes ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//

// Needs to be at the top for <cmath> on Windows.
// See https://stackoverflow.com/a/6563891/321301
#include "util.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include <unordered_set>

#include "Qwerty/IR/QwertyAttributes.h"
#include "Qwerty/IR/QwertyDialect.h"

namespace {
    struct APIntHash {
        auto operator()(llvm::APInt i) const {
            return llvm::hash_value(i);
        }
    };
} // namespace

using namespace qwerty;

#include "Qwerty/IR/QwertyOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Qwerty/IR/QwertyOpsAttributes.cpp.inc"

void QwertyDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "Qwerty/IR/QwertyOpsAttributes.cpp.inc"
    >();
}

void BuiltinBasisAttr::expandToFlatVectors(
        llvm::SmallVectorImpl<qwerty::BasisVectorAttr> &out) const {
    size_t expanded_size = 1ULL << getDim();
    out.reserve(out.size() + expanded_size);
    uint64_t dim = getDim();

    for (size_t i = 0; i < expanded_size; i++) {
        llvm::APInt eigentemp(getDim(), i, /*isSigned=*/false);
        out.push_back(BasisVectorAttr::get(getContext(), getPrimBasis(), eigentemp,
                                           dim, false));
    }
}

uint64_t BasisVectorListAttr::getNumPhases() const {
    uint64_t num_phases = 0;
    for (BasisVectorTreeAttr vec : getVectors()) {
        num_phases += vec.getNumPhases();
    }
    return num_phases;
}

void BasisVectorAttr::print(mlir::AsmPrinter &printer) const {
    if (hasPhase()) {
        printer << "exp(i*theta)*";
    }

    printer << "\"|";
    for (uint64_t i = 0; i < getDim(); i++) {
        size_t off = getDim() - 1 - i;
        unsigned char bit = 0;
        if (getEigenbits()[off]) {
            bit = 1;
        }
        if (!bit) {
            switch (getPrimBasis()) {
            case PrimitiveBasis::X: printer << "p"; break;
            case PrimitiveBasis::Y: printer << "i"; break;
            case PrimitiveBasis::Z: printer << "0"; break;
            default: assert(0 && "Missing PrimitiveBasis case");
            }
        } else {
            switch (getPrimBasis()) {
            case PrimitiveBasis::X: printer << "m"; break;
            case PrimitiveBasis::Y: printer << "j"; break;
            case PrimitiveBasis::Z: printer << "1"; break;
            default: assert(0 && "Missing PrimitiveBasis case");
            }
        }
    }
    printer << ">\"";
}

mlir::Attribute BasisVectorAttr::parse(mlir::AsmParser &parser, mlir::Type odsType) {
    bool hasPhase = false;
    if (!parser.parseOptionalKeyword("exp")) {
        if (parser.parseLParen()
                || parser.parseKeyword("i")
                || parser.parseStar()
                || parser.parseKeyword("theta")
                || parser.parseRParen()
                || parser.parseStar()) {
            return {};
        }
        hasPhase = true;
    }

    std::string ket;
    if (parser.parseString(&ket)) {
        return {};
    }
    if (ket.size() < 3 || ket[0] != '|' || ket[ket.size()-1] != '>') {
        return {};
    }
    // Remove | and > from |010>
    std::string bits = ket.substr(1, ket.size()-2);

    PrimitiveBasis prim_basis = PrimitiveBasis::Z; // Initialize to make compiler happy
    uint64_t dim = bits.size();
    llvm::APInt eigenbits(dim, 0, /*isSigned=*/false);
    for (uint64_t i = 0; i < dim; i++) {
        char bit = bits[i];
        PrimitiveBasis new_prim_basis;
        unsigned char new_bit;
        switch (bit) {
        case 'p': case '+': new_bit = 0; new_prim_basis = PrimitiveBasis::X; break;
        case 'm': case '-': new_bit = 1; new_prim_basis = PrimitiveBasis::X; break;
        case 'i': new_bit = 0; new_prim_basis = PrimitiveBasis::Y; break;
        case 'j': new_bit = 1; new_prim_basis = PrimitiveBasis::Y; break;
        case '0': new_bit = 0; new_prim_basis = PrimitiveBasis::Z; break;
        case '1': new_bit = 1; new_prim_basis = PrimitiveBasis::Z; break;
        default: return {};
        }
        if (!i) {
            prim_basis = new_prim_basis;
        } else if (prim_basis != new_prim_basis) {
            return {};
        }
        size_t off = dim - 1 - i;

        if(new_bit) {
            eigenbits.setBit(off);
        }
    }
    return BasisVectorAttr::get(parser.getContext(),
                                prim_basis, eigenbits, dim, hasPhase);
}

void BasisElemAttr::print(mlir::AsmPrinter &printer) const {
    if (BuiltinBasisAttr std = getStd()) {
        printer << "std:";
        printer.printStrippedAttrOrType(std);
    } else if (BasisVectorListAttr list = getVeclist()) {
        printer << "list:";
        printer.printStrippedAttrOrType(list);
    } else if (ApplyRevolveGeneratorAttr revolve = getRevolve()) {
        printer << "revolve:";
        printer.printStrippedAttrOrType(revolve);
    } else {
        assert(0 && "Invalid basis element state. How did the validator not "
                    "catch this?");
    }
}

mlir::Attribute BasisElemAttr::parse(mlir::AsmParser &parser,
                                     mlir::Type odsType) {
    llvm::StringRef kw;

    if (parser.parseOptionalKeyword(&kw) || parser.parseColon()) {
        return {};
    }

    if (kw == "std") {
        BuiltinBasisAttr std;
        if (parser.parseCustomAttributeWithFallback<BuiltinBasisAttr>(std))
            return {};
        return BasisElemAttr::get(parser.getContext(), std);
    } else if (kw == "list") {
        BasisVectorListAttr list;
        if (parser.parseCustomAttributeWithFallback<BasisVectorListAttr>(list))
            return {};
        return BasisElemAttr::get(parser.getContext(), list);
    } else if (kw == "revolve") {
        ApplyRevolveGeneratorAttr revolve;
        if (parser.parseCustomAttributeWithFallback<ApplyRevolveGeneratorAttr>(revolve))
            return {};
        return BasisElemAttr::get(parser.getContext(), revolve);
    } else {
        return {};
    }
}

uint64_t ApplyRevolveGeneratorAttr::getDim() const {
    // seed is a BasisAttr
    uint64_t seedDim = getSeed().getDim();
    return seedDim + 1;
}

// TODO: This is nonsense and we dislike it.
// Strongly.
PrimitiveBasis ApplyRevolveGeneratorAttr::getPrimBasis() const {
    // Inherit the primitive basis from foo
    return getBv1().getPrimBasis(); // TODO: Is this correct?
}

bool ApplyRevolveGeneratorAttr::isPredicate() const {
  return getSeed().hasPredicate();
}

bool ApplyRevolveGeneratorAttr::hasPhases() const {
  return getSeed().hasPhases() || getBv1().hasPhase() || getBv2().hasPhase();
}

uint64_t ApplyRevolveGeneratorAttr::getNumPhases() const {
    return getSeed().getNumPhases() + getBv1().hasPhase() + getBv2().hasPhase();
}

uint64_t BasisVectorListAttr::getDim() const {
    llvm::ArrayRef<BasisVectorTreeAttr> vectors = getVectors();
    assert(!vectors.empty() && "Empty BasisVectorList. How? The verifier should catch this!");
    return vectors[0].getDim();
}

uint64_t SuperposElemAttr::getDim() const {
    size_t n_qubits = 0;
    for (qwerty::BasisVectorAttr vec : getVectors()) {
        n_qubits += vec.getDim();
    }
    return n_qubits;
}

uint64_t SuperposAttr::getDim() const {
    llvm::ArrayRef<SuperposElemAttr> elems = getElems();
    assert(!elems.empty() && "Empty BasisVectorList. How? The verifier should catch this!");
    return elems[0].getDim();
}

llvm::APInt SuperposElemAttr::getEigenbits() const {
    llvm::APInt eigenbits(/*numBits=*/0UL, /*val=*/0UL, /*isSigned=*/false);
    for (qwerty::BasisVectorAttr vec : getVectors()) {
        eigenbits = eigenbits.concat(vec.getEigenbits());
    }
    return eigenbits;
}

PrimitiveBasis BasisVectorListAttr::getPrimBasis() const {
    llvm::ArrayRef<BasisVectorAttr> vectors = getVectors();
    assert(!vectors.empty() && "Empty BasisVectorList. How? The verifier should catch this!");
    return vectors[0].getPrimBasis();
}

bool BasisVectorListAttr::isPredicate() const {
    return getVectors().size() < (1ULL << getDim());
}

bool BasisVectorListAttr::hasPhases() const {
    for (BasisVectorTreeAttr vec : getVectors()) {
        if (vec.hasPhases()) {
            return true;
        }
    }
    return false;
}

uint64_t BasisElemAttr::getDim() const {
    if (getStd()) {
        return getStd().getDim();
    } else if (getVeclist()) {
        return getVeclist().getDim();
    } else if (getRevolve()) {
        return getRevolve().getDim();
    } else {
        assert(0 && "None of basis, vector list, or revolve generator in this basis element. "
                    "Verifier should catch this!");
        return 0;
    }
}

PrimitiveBasis BasisElemAttr::getPrimBasis() const {
    if (getStd()) {
        return getStd().getPrimBasis();
    } else if (getVeclist()) {
        return getVeclist().getPrimBasis();
    } else if (getRevolve()) {
        return getRevolve().getPrimBasis();
    } else {
        assert(0 && "None of basis, vector list, or revolve generator in this basis element. "
                    "Verifier should catch this!");
        return (PrimitiveBasis)-1;
    }
}

bool BasisElemAttr::isPredicate() const {
    if (getStd()) {
        return false;
    } else if (getVeclist()) {
        return getVeclist().isPredicate();
    } else if (getRevolve()) {
        return getRevolve().isPredicate();
    } else {
        assert(0 && "None of basis, vector list, or revolve generator in this basis element. "
                    "Verifier should catch this!");
        return false;
    }
}

uint64_t BasisElemAttr::getNumPhases() const {
    if (getStd()) {
        return 0;
    } else if (getVeclist()) {
        return getVeclist().getNumPhases();
    } else if (getRevolve()) {
        return getRevolve().getNumPhases();
    } else {
        assert(0 && "None of basis, vector list, or revolve generator in this basis element. "
                    "Verifier should catch this!");
        return 0;
    }
}

bool BasisElemAttr::hasPhases() const {
    if (getStd()) {
        return false;
    } else if (getVeclist()) {
        return getVeclist().hasPhases();
    } else if (getRevolve()) {
        return getRevolve().hasPhases();
    } else {
        assert(0 && "None of basis, vector list, or revolve generator in this basis element. "
                    "Verifier should catch this!");
        return 0;
    }
}

uint64_t BasisAttr::getDim() const {
    llvm::ArrayRef<BasisElemAttr> elems = getElems();
    uint64_t total_dim = 0;
    for (size_t i = 0; i < elems.size(); i++) {
        total_dim += elems[i].getDim();
    }
    return total_dim;
}

bool BasisAttr::hasPredicate() const {
    llvm::ArrayRef<BasisElemAttr> elems = getElems();
    for (size_t i = 0; i < elems.size(); i++) {
        if (elems[i].isPredicate()) {
            return true;
        }
    }
    return false;
}

bool BasisAttr::hasNonPredicate() const {
    llvm::ArrayRef<BasisElemAttr> elems = getElems();
    for (size_t i = 0; i < elems.size(); i++) {
        if (!elems[i].isPredicate()) {
            return true;
        }
    }
    return false;
}

bool BasisAttr::hasPhases() const {
    for (BasisElemAttr elem : getElems()) {
        if (elem.hasPhases()) {
            return true;
        }
    }
    return false;
}

bool BasisAttr::hasOnlyOnes() const {
    if (hasNonPredicate()) {
        // Fast path
        return false;
    }
    // Past this point, every element must be predicates

    for (BasisElemAttr elem : getElems()) {
        BasisVectorListAttr vl = elem.getVeclist();
        assert(vl && !vl.getVectors().empty());
        if (vl.getVectors().size() > 1) {
            // Duplicate vectors are not allowed, so this can't be possible
            return false;
        }
        BasisVectorAttr vec = vl.getVectors()[0];
        // Predicate bases should not have phases (they are meaningless)
        assert(!vec.hasPhase());
        if (vec.getPrimBasis() != PrimitiveBasis::Z
                || !vec.getEigenbits().isAllOnes()) {
            return false;
        }
    }

    return true;
}

uint64_t BasisAttr::getNumPhases() const {
    llvm::ArrayRef<BasisElemAttr> elems = getElems();
    uint64_t total_n_phases = 0;
    for (size_t i = 0; i < elems.size(); i++) {
        total_n_phases += elems[i].getNumPhases();
    }
    return total_n_phases;
}

BasisAttr BasisAttr::getAllOnesBasis(mlir::MLIRContext *ctx, size_t dim) {
    return BasisAttr::get(ctx,
        std::initializer_list<BasisElemAttr>{
            BasisElemAttr::get(ctx,
                BasisVectorListAttr::get(ctx,
                    std::initializer_list<BasisVectorAttr>{
                        BasisVectorAttr::get(ctx,
                            PrimitiveBasis::Z,
                            Eigenstate::MINUS,
                            dim,
                            /*hasPhase=*/false)}))});
}

mlir::LogicalResult BuiltinBasisAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        PrimitiveBasis prim_basis,
        uint64_t dim) {
    if (!dim) {
        return emitError() << "Zero dimension not allowed";
    }
    return mlir::success();
}

mlir::LogicalResult BasisVectorAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        PrimitiveBasis prim_basis,
        mlir::IntegerAttr eigenbitsAttr,
        uint64_t dim,
        bool phase) {
    if (!dim) {
        return emitError() << "Zero dimension not allowed";
    }
    llvm::APInt eigenbits = eigenbitsAttr.getValue();
    if (eigenbits.getBitWidth() > dim) {
        return emitError() << "Invariant for APInt version of eigenbits: "
                            << eigenbits.getBitWidth() 
                            << " vs "
                            << dim;
    }
    return mlir::success();
}

mlir::LogicalResult BasisVectorListAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        llvm::ArrayRef<BasisVectorTreeAttr> vectors) {
    if (vectors.empty()) {
        return emitError() << "List of vectors cannot be empty";
    }

    // First, check dimensions
    uint64_t dim = 0;
    for (auto it = vectors.begin(); it != vectors.end(); it++) {
        const BasisVectorTreeAttr &vec = *it;
        uint64_t new_dim = vec.getDim();
        if (it == vectors.begin()) {
            dim = new_dim;
        } else {
            if (dim != new_dim) {
                return emitError() << "Vector dimension mismatch: " << dim
                                   << " != " << new_dim;
            }
        }
    }
    llvm::DenseSet<BasisVectorTreeAttr> seen;
    for (BasisVectorTreeAttr vec : vectors) {
        if (!seen.insert(vec).second) {
            return emitError() << "Basis vector already seen";
        }
    }
    return mlir::success();
}

mlir::LogicalResult BasisVectorTreeAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        BasisVectorTreeKind kind,
        mlir::FloatAttr tilt,
        llvm::ArrayRef<BasisVectorTreeAttr> children) {
    bool hasTilt = (bool)tilt;
    size_t n = children.size();

    auto tiltErr = [&]() {
        return emitError() << stringifyBasisVectorTreeKind(kind)
                           << " must not carry an angle";
    };
    auto arityErr = [&](const char *numChildren) {
        return emitError() << stringifyBasisVectorTreeKind(kind) << " expects "
                           << numChildren << " child(ren), got " << n;
    };

    switch (kind) {
    case BasisVectorTreeKind::ZeroVector:
    case BasisVectorTreeKind::OneVector:
    case BasisVectorTreeKind::PadVector:
    case BasisVectorTreeKind::TargetVector:
    case BasisVectorTreeKind::VectorUnit:
        if (hasTilt) return tiltErr();
        if (n != 0) return arityErr("no");
        return mlir::success();

    case BasisVectorTreeKind::VectorTilt:
        if (n != 1) return arityErr("exactly 1");
        return mlir::success();

    case BasisVectorTreeKind::UniformVectorSuperpos:
        if (hasTilt) return tiltErr();
        if (n != 2) return arityErr("exactly 2");
        if (children[0].getDim() != children[1].getDim()) {
            return emitError() << "Left child and Right child must be equivalent in dimension, recieved" << "l:"
                                    << children[0].getDim() << "and r:" << children[1].getDim();
        }
        return mlir::success();

    case BasisVectorTreeKind::VectorTensor:
        if (hasTilt) return tiltErr();
        if (n < 2) return arityErr("at least 2");
        return mlir::success();
    }
    return emitError() << "unknown BasisVectorTreeKind";
}

namespace {

double tiltDegToRad(double angle_deg) {
    return angle_deg / 360.0 * 2.0 * M_PI;
}

// Canonicalizes an angle in radians into [0, 2*pi).
double canonRad(double theta) {
    double two_pi = 2.0 * M_PI;
    double m = std::fmod(theta, two_pi);
    if (m < 0.0) {
        m += two_pi;
    }
    return m;
}

// Returns true iff theta is approximately angle
bool radsApproxEqual(double theta, double angle) {
    double diff = canonRad(theta - angle);
    return diff < ATOL || (2.0 * M_PI) - diff < ATOL;
}

} // namespace

std::optional<std::pair<qwerty::BasisVectorAttr, double>>
BasisVectorTreeAttr::tryFlatten() const {
    mlir::MLIRContext *ctx = getContext();

    switch (getKind()) {
    case BasisVectorTreeKind::ZeroVector:
        return std::make_pair(
            BasisVectorAttr::get(ctx, PrimitiveBasis::Z, llvm::APInt(1, 0), 1,false), 0.0); // hasPhase = false
    case BasisVectorTreeKind::OneVector:
        return std::make_pair(
            BasisVectorAttr::get(ctx, PrimitiveBasis::Z, llvm::APInt(1, 1), 1, false), 0.0); // hasPhase = false

    case BasisVectorTreeKind::PadVector:
    case BasisVectorTreeKind::TargetVector:
    case BasisVectorTreeKind::VectorUnit:
        return std::nullopt;

    case BasisVectorTreeKind::VectorTilt: {
        mlir::FloatAttr tilt = getTilt();
        if (!tilt) {
            // Dynamic tilt: the angle is a runtime phases() operand, so there
            // is no compile-time flat form.
            return std::nullopt;
        }
        auto sub = getChildren()[0].tryFlatten();
        if (!sub) {
            return std::nullopt;
        }
        return std::make_pair(
            sub->first, sub->second + tiltDegToRad(tilt.getValueAsDouble()));
    }

    case BasisVectorTreeKind::UniformVectorSuperpos: {
        auto lhs = getChildren()[0].tryFlatten();
        auto rhs = getChildren()[1].tryFlatten();
        if (!lhs || !rhs) {
            return std::nullopt;
        }
        BasisVectorAttr lv = lhs->first;
        BasisVectorAttr rv = rhs->first;
        // for Bell States
        if (lv.getPrimBasis() != PrimitiveBasis::Z
                || rv.getPrimBasis() != PrimitiveBasis::Z
                || lv.getDim() != 1 || rv.getDim() != 1
                || lv.getEigenbits() == rv.getEigenbits()) {
            return std::nullopt;
        }
        // Order the terms so r0/r1 are the phases on the |0>/|1> components
        double r0 = lhs->second;
        double r1 = rhs->second;
        if (lv.getEigenbits().isAllOnes()) {
            std::swap(r0, r1);
        }
        double rel = r1 - r0;

        PrimitiveBasis prim_basis;
        unsigned eigenbit;
        if (radsApproxEqual(rel, 0.0)) {                      // |0> + |1>
            prim_basis = PrimitiveBasis::X;
            eigenbit = 0;
        } else if (radsApproxEqual(rel, M_PI)) {              // |0> - |1>
            prim_basis = PrimitiveBasis::X;
            eigenbit = 1;
        } else if (radsApproxEqual(rel, M_PI / 2.0)) {        // |0> + i|1>
            prim_basis = PrimitiveBasis::Y;
            eigenbit = 0;
        } else if (radsApproxEqual(rel, 3.0 * M_PI / 2.0)) {  // |0> - i|1>
            prim_basis = PrimitiveBasis::Y;
            eigenbit = 1;
        } else {
            // non-Pauli superposition: stays a tree.
            return std::nullopt;
        }
        return std::make_pair(BasisVectorAttr::get(ctx, prim_basis, llvm::APInt(1, eigenbit), 1, false), r0);
    }

    case BasisVectorTreeKind::VectorTensor: {
        std::optional<PrimitiveBasis> prim_basis;
        llvm::APInt eigenbits = llvm::APInt::getZero(0);
        uint64_t dim = 0;
        double residual = 0.0;
        for (BasisVectorTreeAttr child : getChildren()) {
            auto flat = child.tryFlatten();
            if (!flat) {
                return std::nullopt;
            }
            if (prim_basis && *prim_basis != flat->first.getPrimBasis()) {
                // Mixed primitive bases do not flatten to a single vector.
                return std::nullopt;
            }
            prim_basis = flat->first.getPrimBasis();
            // Leftmost factor lands in the most significant bits, matching
            // the eigenbit convention documented on BasisVectorAttr.
            eigenbits = eigenbits.concat(flat->first.getEigenbits());
            dim += flat->first.getDim();
            residual += flat->second;
        }
        return std::make_pair(BasisVectorAttr::get(ctx, *prim_basis, eigenbits, dim, false), residual);
    }
    }
    llvm_unreachable("unknown BasisVectorTreeKind");
}

uint64_t BasisVectorTreeAttr::getDim() const {
    switch (getKind()) {
    case BasisVectorTreeKind::ZeroVector:
    case BasisVectorTreeKind::OneVector:
    case BasisVectorTreeKind::PadVector:
    case BasisVectorTreeKind::TargetVector:
        return 1;
    case BasisVectorTreeKind::VectorUnit:
        return 0;
    case BasisVectorTreeKind::VectorTilt:
    case BasisVectorTreeKind::UniformVectorSuperpos:
        return getChildren()[0].getDim();
    case BasisVectorTreeKind::VectorTensor: {
        uint64_t dim = 0;
        for (BasisVectorTreeAttr child : getChildren()) {
            dim += child.getDim();
        }
        return dim;
    }
    }
    llvm_unreachable("unknown BasisVectorTreeKind");

}


bool BasisVectorTreeAttr::hasPhases() const {
    switch (getKind()) {
    case BasisVectorTreeKind::ZeroVector:
    case BasisVectorTreeKind::OneVector:
    case BasisVectorTreeKind::PadVector:
    case BasisVectorTreeKind::TargetVector:
    case BasisVectorTreeKind::VectorUnit:
        return false;
    case BasisVectorTreeKind::VectorTilt:
        return true;
    case BasisVectorTreeKind::UniformVectorSuperpos:
    case BasisVectorTreeKind::VectorTensor:
        for (BasisVectorTreeAttr child : getChildren()) {
            if (child.hasPhases()) {
                return true;
            }
        }
        return false;
    }
    llvm_unreachable("unknown BasisVectorTreeKind");
}

uint64_t BasisVectorTreeAttr::getNumPhases() const {
    switch (getKind()) {
    case BasisVectorTreeKind::ZeroVector:
    case BasisVectorTreeKind::OneVector:
    case BasisVectorTreeKind::PadVector:
    case BasisVectorTreeKind::TargetVector:
    case BasisVectorTreeKind::VectorUnit:
        return 0;
    case BasisVectorTreeKind::VectorTilt: {
        uint64_t self = getTilt() ? 0 : 1;
        return self + getChildren()[0].getNumPhases();
    }
    case BasisVectorTreeKind::UniformVectorSuperpos:
    case BasisVectorTreeKind::VectorTensor: {
        uint64_t total = 0;
        for (BasisVectorTreeAttr child : getChildren()) {
            total += child.getNumPhases();
        }
        return total;
    }
    }
    llvm_unreachable("unknown BasisVectorTreeKind");
}

void BasisVectorTreeAttr::print(mlir::AsmPrinter &printer) const {
    printer << "<" << stringifyBasisVectorTreeKind(getKind());
    if (mlir::FloatAttr tilt = getTilt()) {
        // Constant tilt
        printer << " tilt " << tilt;
    } else if (getKind() == BasisVectorTreeKind::VectorTilt) {
        // Dynamic tilt
        printer << " tilt theta";
    }
    printer << " [";
    llvm::ArrayRef<BasisVectorTreeAttr> children = getChildren();
    for (size_t i = 0; i < children.size(); i++) {
        if (i) {
            printer << ", ";
        }
        printer << children[i];
    }
    printer << "]>";
}

mlir::Attribute BasisVectorTreeAttr::parse(mlir::AsmParser &parser, mlir::Type odsType) {
    llvm::SMLoc loc = parser.getCurrentLocation();
    if (parser.parseLess()) {
        return {};
    }

    llvm::StringRef kindKeyword;
    if (parser.parseKeyword(&kindKeyword)) {
        return {};
    }
    std::optional<BasisVectorTreeKind> kind =
        symbolizeBasisVectorTreeKind(kindKeyword);
    if (!kind) {
        parser.emitError(loc, "unknown BasisVectorTreeKind: '")
            << kindKeyword << "'";
        return {};
    }

    mlir::FloatAttr tilt;
    if (succeeded(parser.parseOptionalKeyword("tilt"))) {
        if (failed(parser.parseOptionalKeyword("theta"))) {
            if (parser.parseAttribute(tilt)) {
                return {};
            }
        }
    }

    llvm::SmallVector<BasisVectorTreeAttr> children;
    if (parser.parseLSquare()) {
        return {};
    }
    if (failed(parser.parseOptionalRSquare())) {
        do {
            BasisVectorTreeAttr child;
            if (parser.parseAttribute(child)) {
                return {};
            }
            children.push_back(child);
        } while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRSquare()) {
            return {};
        }
    }

    if (parser.parseGreater()) {
        return {};
    }

    return parser.getChecked<BasisVectorTreeAttr>(
        loc, parser.getContext(), *kind, tilt, children);
}


mlir::LogicalResult ApplyRevolveGeneratorAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        BasisAttr foo,
        BasisVectorAttr bv1,
        BasisVectorAttr bv2) {
    if (bv1.getDim() != 1 || bv2.getDim() != 1) {
        return emitError() << "Basis vectors we are using to revolve around must have dim 1";
    }

    // TODO: We also need to check if foo fully spans

    return mlir::success();
}

mlir::LogicalResult SuperposElemAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        mlir::FloatAttr prob_attr,
        mlir::FloatAttr phase_attr,
        llvm::ArrayRef<BasisVectorAttr> vectors) {
    double prob = prob_attr.getValueAsDouble();
    if (prob < ATOL) {
        return emitError() << "Probability must be nonzero";
    }
    if (vectors.empty()) {
        return emitError() << "Empty list of vectors is not allowed";
    }
    return mlir::success();
}

mlir::LogicalResult SuperposAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        llvm::ArrayRef<SuperposElemAttr> elems) {
    if (elems.empty()) {
        return emitError() << "Empty list of superpos elems is not allowed";
    }

    bool first = true;
    double sum = 0.0;
    size_t last_dim;
    llvm::ArrayRef<BasisVectorAttr> last_vecs;
    std::unordered_set<llvm::APInt, APIntHash> eigenbits_seen;

    for (SuperposElemAttr elem : elems) {
        sum += elem.getProb().getValueAsDouble();

        size_t this_dim = elem.getDim();
        llvm::ArrayRef<BasisVectorAttr> this_vecs = elem.getVectors();
        if (first) {
            last_dim = this_dim;
            last_vecs = this_vecs;
        } else {
            if (last_dim != this_dim) {
                return emitError() << "Element dimensions do not match up: "
                                   << last_dim << " != " << this_dim;
            }
            if (last_vecs.size() != this_vecs.size()) {
                return emitError() << "Basis vectors lists must have the "
                                   << "same size:"
                                   << last_vecs.size() << " != "
                                   << this_vecs.size();
            }
            for (auto [last_vec, this_vec] : llvm::zip(last_vecs, this_vecs)) {
                // TODO: remove this check? works well enough for now though
                if (last_vec.getDim() != this_vec.getDim()) {
                    return emitError() << "Basis vector dimension does not "
                                       << "match: " << last_vec.getDim()
                                       << " != " << this_vec.getDim();
                }

                if (last_vec.getPrimBasis() != this_vec.getPrimBasis()) {
                    return emitError()
                        << "Basis vector primitive basis mismatch: "
                        << stringifyPrimitiveBasis(last_vec.getPrimBasis())
                        << " != "
                        << stringifyPrimitiveBasis(this_vec.getPrimBasis());
                }
            }
        }

        first = false;
    }

    return mlir::success();
}

mlir::LogicalResult BasisElemAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        BuiltinBasisAttr std,
        BasisVectorListAttr list,
        ApplyRevolveGeneratorAttr revolve) {
    if (!!std ^ !!list ^ !!revolve) {
        return mlir::success();
    } else {
        return emitError() << "A standard basis xor a basis vector list xor a revolve generator are "
                              "required for a basis element";
    }
}
