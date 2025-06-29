//===- QCircDialect.td - QCirc dialect -----------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_DIALECT_TD
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// QCirc dialect definition.
//===----------------------------------------------------------------------===//

def QCirc_Dialect : Dialect {
    let name = "qcirc";
    let summary = "Dialect for quantum circuit operations";
    let description = [{
        Low-level value-semantics gate dialect
    }];
    let cppNamespace = "::qcirc";

  let useDefaultTypePrinterParser = 1; // can change this to 0 later to change the type printer/parser
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    private:
      void registerTypes();
      void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class QCirc_Op<string mnemonic, list<Trait> traits = []> :
        Op<QCirc_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base quantum attribute definition.
//===----------------------------------------------------------------------===//

class QCirc_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
        AttrDef<QCirc_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

include "QCirc/IR/QCircTypes.td"
// TODO: fix the include structure?

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_DIALECT_TD
