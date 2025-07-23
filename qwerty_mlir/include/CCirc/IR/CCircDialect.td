//===- CCircDialect.td - CCirc dialect -----------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_DIALECT_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// CCirc dialect definition.
//===----------------------------------------------------------------------===//

def CCirc_Dialect : Dialect {
    let name = "ccirc";
    let summary = "Dialect for classical circuits/logic";
    let description = [{
        Ops for a classical netlist containing typical gates such as NOT, XOR, AND, etc.
    }];
    let cppNamespace = "::ccirc";

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

class CCirc_Op<string mnemonic, list<Trait> traits = []> :
        Op<CCirc_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base quantum attribute definition.
//===----------------------------------------------------------------------===//

class CCirc_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
        AttrDef<CCirc_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

include "CCirc/IR/CCircTypes.td"
// TODO: fix the include structure?

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_DIALECT_TD
