//===- QCircInterfaces.h - QCirc dialect interfaces/traits --*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_QCIRC_IR_QCIRC_INTERFACES_H
#define DIALECT_INCLUDE_QCIRC_IR_QCIRC_INTERFACES_H

#include "mlir/IR/OpDefinition.h" // for TraitBase
#include "mlir/IR/PatternMatch.h" // for RewriterBase

namespace qcirc {

// Does not directly affect qubits. Specifically, when taking the adjoint of
// something, you can ignore this op entirely
template <typename ConcreteType>
class IsStationaryOpTrait : public mlir::OpTrait::TraitBase<ConcreteType, IsStationaryOpTrait> {};

} // namespace qcirc

#include "QCirc/IR/QCircOpsTypeInterfaces.h.inc"
#include "QCirc/IR/QCircOpsOpInterfaces.h.inc"

#endif // DIALECT_INCLUDE_QCIRC_IR_QCIRC_INTERFACES_H
