//===- QwertyAttributes.cpp - Qwerty dialect attributes ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/APInt.h"
#include <unordered_set>

#include "Qwerty/IR/QwertyAttributes.h"
#include "Qwerty/IR/QwertyDialect.h"

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

qwerty::BasisVectorListAttr BuiltinBasisAttr::expandToVeclist() const {
    llvm::SmallVector<qwerty::BasisVectorAttr> vectors;
    size_t expanded_size = 1ULL << getDim();
    vectors.reserve(expanded_size);
    uint64_t dim = getDim();

    for (size_t i = 0; i < expanded_size; i++) {
        llvm::APInt eigentemp(getDim(), i);
        vectors.push_back(BasisVectorAttr::get(getContext(), getPrimBasis(), eigentemp,
                                               dim, false));
    }

    return BasisVectorListAttr::get(getContext(), vectors);
}

uint64_t BasisVectorListAttr::getNumPhases() const {
    uint64_t num_phases = 0;
    for (BasisVectorAttr vec : getVectors()) {
        if (vec.hasPhase()) {
            num_phases++;
        }
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
    llvm::APInt eigenbits(dim, 0);
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
    if (getStd()) {
        printer << "std:";
        printer.printStrippedAttrOrType(getStd());
    } else if (getVeclist()) {
        printer << "list:";
        printer.printStrippedAttrOrType(getVeclist());
    } else {
        assert(0 && "Invalid basis element state. How did the validator not "
                    "catch this?");
    }
}

mlir::Attribute BasisElemAttr::parse(mlir::AsmParser &parser, mlir::Type odsType) {
    llvm::StringRef kw;
    if (parser.parseKeyword(&kw)
        || parser.parseColon()) {
        return {};
    }

    if (kw == "std") {
        BuiltinBasisAttr std;
        if (parser.parseCustomAttributeWithFallback<BuiltinBasisAttr>(std)) {
            return {};
        }
        return BasisElemAttr::get(parser.getContext(), std);
    } else if (kw == "list") {
        BasisVectorListAttr list;
        if (parser.parseCustomAttributeWithFallback<BasisVectorListAttr>(list)) {
            return {};
        }
        return BasisElemAttr::get(parser.getContext(), list);
    } else {
        return {};
    }
}

uint64_t BasisVectorListAttr::getDim() const {
    llvm::ArrayRef<BasisVectorAttr> vectors = getVectors();
    assert(!vectors.empty() && "Empty BasisVectorList. How? The verifier should catch this!");
    return vectors[0].getDim();
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
    for (BasisVectorAttr vec : getVectors()) {
        if (vec.hasPhase()) {
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
    } else {
        assert(0 && "Neither basis nor vector list in this basis element. "
                    "Verifier should catch this!");
        return 0;
    }
}

PrimitiveBasis BasisElemAttr::getPrimBasis() const {
    if (getStd()) {
        return getStd().getPrimBasis();
    } else if (getVeclist()) {
        return getVeclist().getPrimBasis();
    } else {
        assert(0 && "Neither basis nor vector list in this basis element. "
                    "Verifier should catch this!");
        return (PrimitiveBasis)-1;
    }
}

bool BasisElemAttr::isPredicate() const {
    if (getStd()) {
        return false;
    } else if (getVeclist()) {
        return getVeclist().isPredicate();
    } else {
        assert(0 && "Neither basis nor vector list in this basis element. "
                    "Verifier should catch this!");
        return false;
    }
}

uint64_t BasisElemAttr::getNumPhases() const {
    if (getStd()) {
        return 0;
    } else if (getVeclist()) {
        return getVeclist().getNumPhases();
    } else {
        assert(0 && "Neither basis nor vector list in this basis element. "
                    "Verifier should catch this!");
        return 0;
    }
}

bool BasisElemAttr::hasPhases() const {
    if (getStd()) {
        return false;
    } else if (getVeclist()) {
        return getVeclist().hasPhases();
    } else {
        assert(0 && "Neither basis nor vector list in this basis element. "
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

BasisVectorAttr BasisVectorAttr::deletePhase() const {
    if (!hasPhase()) {
        return *this;
    }
    return BasisVectorAttr::get(
        getContext(), getPrimBasis(), getEigenbits(), getDim(),
        /*hasPhase=*/false);
}

BasisVectorListAttr BasisVectorListAttr::deletePhases() const {
    if (!hasPhases()) {
        return *this;
    }

    llvm::SmallVector<BasisVectorAttr> vecs;
    vecs.reserve(getVectors().size());

    for (BasisVectorAttr vec : getVectors()) {
        vecs.push_back(vec.deletePhase());
    }

    return BasisVectorListAttr::get(getContext(), vecs);
}

BasisElemAttr BasisElemAttr::deletePhases() const {
    if (getStd() || !hasPhases()) {
        return *this;
    }

    return BasisElemAttr::get(
        getContext(), getVeclist().deletePhases());
}

BasisAttr BasisAttr::deletePhases() const {
    if (!hasPhases()) {
        return *this;
    }

    llvm::SmallVector<BasisElemAttr> elems;
    elems.reserve(getElems().size());

    for (BasisElemAttr elem : getElems()) {
        elems.push_back(elem.deletePhases());
    }

    return BasisAttr::get(getContext(), elems);
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
        llvm::ArrayRef<BasisVectorAttr> vectors) {
    if (vectors.empty()) {
        return emitError() << "List of vectors cannot be empty";
    }

    // First, check dimensions
    uint64_t dim = 0;
    PrimitiveBasis prim_basis = PrimitiveBasis::Z; // Initialize to make compiler happy
    for (auto it = vectors.begin(); it != vectors.end(); it++) {
        const BasisVectorAttr &vec = *it;
        uint64_t new_dim = vec.getDim();
        PrimitiveBasis new_prim_basis = vec.getPrimBasis();
        if (it == vectors.begin()) {
            dim = new_dim;
            prim_basis = new_prim_basis;
        } else {
            if (dim != new_dim) {
                return emitError() << "Vector dimension mismatch: " << dim
                                   << " != " << new_dim;
            }
            if (prim_basis != new_prim_basis) {
                return emitError() << "PrimitiveBasis mismatch: "
                                   << stringifyPrimitiveBasis(prim_basis)
                                   << " != " << stringifyPrimitiveBasis(new_prim_basis);
            }
        }
    }

    struct APIntHash {
        auto operator()(llvm::APInt i) const {
            return llvm::hash_value(i);
        }
    };
    std::unordered_set<llvm::APInt, APIntHash> basisVectorsSeen;
    for (auto it = vectors.begin(); it != vectors.end(); it++) {
        const BasisVectorAttr &vec = *it;
        llvm::APInt eigenbits = vec.getEigenbits();
        if (eigenbits.getBitWidth() > dim) {
            return emitError() << "Invariant for APInt version of eigenbits Vector List: "
                                << eigenbits.getBitWidth() 
                                << " vs "
                                << dim;
        }

        if (basisVectorsSeen.count(eigenbits) != 0) {
            return emitError() << "Basis vector already seen";
        }
        basisVectorsSeen.insert(eigenbits);
    }
    return mlir::success();
}

mlir::LogicalResult BasisElemAttr::verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
        BuiltinBasisAttr std,
        BasisVectorListAttr list) {
    if (!!std ^ !!list) {
        return mlir::success();
    } else {
        return emitError() << "A standard basis xor a basis vector list are "
                              "required for a basis element";
    }
}
