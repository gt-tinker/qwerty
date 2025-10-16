//===- QwertyOps.td - Qwerty dialect attributes ------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_ATTRIBUTES_TD
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_ATTRIBUTES_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "Qwerty/IR/QwertyDialect.td"

// Paulis have a -1 and a +1 eigenvalue. For example, Z|0⟩ = (+1)|0⟩, and Z|1⟩
// = (-1)|1⟩. Similarly, X|+⟩ = (+1)|+⟩, and X|-⟩ = (-1)|-⟩. We identify '+' as
// (X, PLUS), and '1' as (Z, MINUS), and so on. See Section 2.1 of the CGO
// paper.
def Qwerty_EigenstateAttr : I64EnumAttr<
        "Eigenstate", "Eigenstate to prepare",
        [
            // The order here MUST match the ordering in defs.hpp or
            // eigenstate_to_qwerty() will break
            I64EnumAttrCase<"PLUS", 0>,
            I64EnumAttrCase<"MINUS", 1>,
        ]> {
    let cppNamespace = "::qwerty";
}

// The three Paulis plus the Fourier basis. Used above to identify qubit
// literals and bases. Fourier is here because of fourier[N]. See Section 2.2
// of the CGO paper.
def Qwerty_PrimitiveBasisAttr : I64EnumAttr<
        "PrimitiveBasis", "Operator whose eigenstates to prepare",
        [
            I64EnumAttrCase<"X", 0>,
            I64EnumAttrCase<"Y", 1>,
            I64EnumAttrCase<"Z", 2>,
            I64EnumAttrCase<"BELL", 3>,
        ]> {
    let cppNamespace = "::qwerty";
}

def Qwerty_BuiltinBasisAttr : Qwerty_Attr<"BuiltinBasis", "std"> {
    let summary = "Pair of primitive basis and dimension";
    let description = [{
        A pair of primitive basis and dimension, like pm[3] or fourier[4]. See
        Section 2.2 of the CGO paper.
    }];
    let parameters = (ins "PrimitiveBasis":$prim_basis, "uint64_t":$dim);
    let assemblyFormat = "$prim_basis `[` $dim `]`";
    let genVerifyDecl = 1;
    let extraClassDeclaration = [{
        qwerty::BasisVectorListAttr expandToVeclist() const;
    }];
}

def Qwerty_ApplyRevolveGeneratorAttr : Qwerty_Attr<"ApplyRevolveGenerator", "revolve"> {
    let summary = "Represents a revolve basis generator";
    let description = [{
    Represents the revolve basis generator of the form 
    foo // {bv1, bv2}.revolve, for use in synthesis of basis generator circuits
    }];
    let parameters = (ins "BasisAttr":$foo, "BasisVectorAttr":$bv1, "BasisVectorAttr":$bv2);
    let assemblyFormat = "$foo `by` ` ` `{` $bv1 `,` $bv2 `}`";
    let genVerifyDecl = 1;
    let extraClassDeclaration = [{
        uint64_t getDim() const;
        qwerty::PrimitiveBasis getPrimBasis() const;
        bool isPredicate() const;
        bool hasPhases() const;
        uint64_t getNumPhases() const;
    }];
}

def Qwerty_BasisVectorAttr : Qwerty_Attr<"BasisVector", "vec"> {
    let summary = "A basis vector for a basis literal";
    let description = [{
        Representation of a basis vector in a basis literal. (See Section 2.2 of
        the CGO paper for the definition of a basis vector.)

        The eigenbits correspond to each bit-position of the vector, where 0
        and 1 correspond to PLUS and MINUS eigenstates respectively, and the
        most significant bit is the leftmost position.

        The hasPhase flag indicates the basis vector has a phase factor e^(iθ).
        The θ is usually an mlir::Value, i.e., a runtime-calculated value. Such
        mlir::Values are passed as operands to operations with basis operands.
    }];
    let parameters = (ins "PrimitiveBasis":$prim_basis,
                          "mlir::IntegerAttr":$eigenbitsAttr,
                          "uint64_t":$dim,
                          "bool":$hasPhase);
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
    let builders = [
        // Useful for lowering QubitLiteral AST nodes
        AttrBuilder<(ins "PrimitiveBasis":$prim_basis, "Eigenstate":$eigenstate, "uint64_t":$dim, "bool":$hasPhase), [{
            llvm::APInt eigenbits(dim, 0);
            switch (eigenstate) {
            case qwerty::Eigenstate::PLUS: eigenbits.clearAllBits(); break;
            case qwerty::Eigenstate::MINUS: eigenbits.setAllBits(); break;
            default: assert(0 && "Unhandled Eigenstate in BasisVectorAttr::build()");
            }
            return get($_ctxt, prim_basis, eigenbits, dim, hasPhase);
        }]>,
        // Convenient to pass an APInt directly instead of having to create an
        // IntegerAttr.
        AttrBuilder<(ins "PrimitiveBasis":$prim_basis, "llvm::APInt":$apint, "uint64_t":$dim, "bool":$hasPhase), [{
            auto type = mlir::IntegerType::get($_ctxt, apint.getBitWidth(), mlir::IntegerType::Unsigned);
            auto intattr = mlir::IntegerAttr::get(type, apint);
            return $_get($_ctxt, prim_basis, intattr, dim, hasPhase);
        }]>
    ];
    let extraClassDeclaration = [{
        bool hasPhase() const { return getHasPhase(); }
        llvm::APInt getEigenbits() const { return getEigenbitsAttr().getValue(); }
    }];
}

def Qwerty_SuperposElemAttr : Qwerty_Attr<"SuperposElem", "superposelem"> {
    let summary = "An element of a superposition";
    let description = [{
        A qubit literal with a probability, intended for use as a superposition
        element. Because this element may consist of multiple primitive bases,
        this attribute contains a sequence of basis vectors. Unlike a basis
        vector, though, the phase of this attribute must be constant.
    }];
    let parameters = (ins "mlir::FloatAttr":$prob, "mlir::FloatAttr":$phase, ArrayRefParameter<"BasisVectorAttr", "">:$vectors);
    let assemblyFormat = "$prob `*` $vectors `at` `tilt` $phase";
    let extraClassDeclaration = [{
        llvm::APInt getEigenbits() const;
        uint64_t getDim() const;
        bool isPredicate() const;
        bool hasPhases() const;
        uint64_t getNumPhases() const;
    }];
    let genVerifyDecl = 1;
}

def Qwerty_SuperposAttr: Qwerty_Attr<"Superpos", "superpos"> {
    let summary = "A set of superposition elements";
    let description = [{
        An attribute holding a superposition to be prepared. Among
        superposition elements, eigenbits must be distinct, dimensions must
        match, and primitive bases must match otherwise. The sum of
        probabilities of superposition elements must also be 1.
    }];
    let parameters = (ins ArrayRefParameter<"SuperposElemAttr", "">:$elems);
    let assemblyFormat = "`(` $elems `)`";
    let extraClassDeclaration = [{
        uint64_t getDim() const;
    }];
    let genVerifyDecl = 1;
}

def Qwerty_BasisVectorListAttr : Qwerty_Attr<"BasisVectorList", "veclist"> {
    let summary = "A basis literal";
    let description = [{
        A representation of a basis literal as an ordered sequence of basis
        vectors. (See Section 2.2 of the CGO paper for the definition of a
        basis literal.)

        The verifier checks that all vectors must have the same dimension,
        same primitive basis, and distinct eigenbits.
    }];
    let parameters = (ins ArrayRefParameter<"BasisVectorAttr", "">:$vectors);
    let assemblyFormat = "`{` $vectors `}`";
    let extraClassDeclaration = [{
        uint64_t getDim() const;
        qwerty::PrimitiveBasis getPrimBasis() const;
        bool isPredicate() const;
        bool hasPhases() const;
        uint64_t getNumPhases() const;
    }];
    let genVerifyDecl = 1;
}

def Qwerty_BasisElemAttr : Qwerty_Attr<"BasisElem", "basiselem"> {
    let summary = "Union of BuiltinBasisAttr, BasisVectorListAttr, and ApplyRevolveGeneratorAttr";
    let description = [{
        A basis element as defined in Section 2.2 of the CGO paper: either a
        built-in basis (e.g., pm[5]), a basis literal (e.g., {'0', '1'}),
        or (not in CGO paper) a revolve construct (foo // {bv1, bv2}.revolve).
    }];
    let parameters = (ins OptionalParameter<"BuiltinBasisAttr">:$std, OptionalParameter<"BasisVectorListAttr">:$veclist, OptionalParameter<"ApplyRevolveGeneratorAttr">:$revolve);
    let hasCustomAssemblyFormat = 1;
    let builders = [
        AttrBuilder<(ins "BuiltinBasisAttr":$std), [{
            return $_get($_ctxt, std, nullptr, nullptr);
        }]>,
        AttrBuilder<(ins "BasisVectorListAttr":$veclist), [{
            return $_get($_ctxt, nullptr, veclist, nullptr);
        }]>,
        AttrBuilder<(ins "ApplyRevolveGeneratorAttr":$revolve), [{
            return $_get($_ctxt, nullptr, nullptr, revolve);
        }]>
    ];
    let extraClassDeclaration = [{
        uint64_t getDim() const;
        qwerty::PrimitiveBasis getPrimBasis() const;
        bool isPredicate() const;
        bool hasPhases() const;
        uint64_t getNumPhases() const;
    }];
    let genVerifyDecl = 1;
}

def Qwerty_BasisAttr : Qwerty_Attr<"Basis", "basis"> {
    let summary = "A basis";
    let description = [{
        A basis in canon form (defined in Section 2.2 of the CGO paper), that
        is, an ordered sequence of basis elements.
    }];
    let parameters = (ins ArrayRefParameter<"BasisElemAttr", "">:$elems);
    let assemblyFormat = "`{` $elems `}`";
    let extraClassDeclaration = [{
        uint64_t getDim() const;
        bool hasPredicate() const;
        bool hasNonPredicate() const;
        bool hasPhases() const;
        bool hasOnlyOnes() const;
        uint64_t getNumPhases() const;
        static qwerty::BasisAttr getAllOnesBasis(mlir::MLIRContext *ctx, size_t dim);
    }];
}

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_ATTRIBUTES_TD
