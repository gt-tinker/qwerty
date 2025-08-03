//===- QwertyDialect.td - Qwerty dialect -----------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QWERTY_IR_QWERTY_DIALECT_TD
#define DIALECT_INCLUDE_QWERTY_IR_QWERTY_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Qwerty dialect definition.
//===----------------------------------------------------------------------===//

def Qwerty_Dialect : Dialect {
    let name = "qwerty";
    let summary = "Dialect corresponding closely to Qwerty code";
    let description = [{
        Address-semantics dialect corresponding closely to Python ASTs calling Qwerty code
    }];
    let cppNamespace = "::qwerty";

  // If you have no types defined in your dialect, this will cause a linker error
  let useDefaultTypePrinterParser = 1; // can change this to 0 later to change the type printer/parser
  // Catastrophic segfaults without this
  let useDefaultAttributePrinterParser = 1;

  // This depends on qcirc because QBundlePackOp / QBundleUnpackOp use the
  // qubit type, and some canonicalization patterns work with qubits directly.
  // This dialect also depends on ccirc because of ops for embedding classical
  // functions.
  let dependentDialects = ["qcirc::QCircDialect", "ccirc::CCircDialect"];

  let extraClassDeclaration = [{
    private:
      void registerTypes();
      void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Qwerty operation definition.
//===----------------------------------------------------------------------===//

class Qwerty_Op<string mnemonic, list<Trait> traits = []> :
        Op<Qwerty_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base Qwerty attribute definition.
//===----------------------------------------------------------------------===//

class Qwerty_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
        AttrDef<Qwerty_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

#endif // DIALECT_INCLUDE_QWERTY_IR_QWERTY_DIALECT_TD
