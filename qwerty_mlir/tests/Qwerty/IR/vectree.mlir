// RUN: qwerty-opt -split-input-file %s | qwerty-opt -split-input-file | FileCheck --strict-whitespace %s
//
// Happy-path round-trip (parse -> print -> parse -> print) of
// BasisVectorTreeAttr. If any case were malformed, parsing would fail and the round-trip would error here.


// A bare leaf atom: '0'  (0 children -> empty brackets)
// CHECK: qwerty.t = #qwerty.vectree<ZeroVector []>
module attributes {qwerty.t = #qwerty.vectree<ZeroVector []>} {}

// -----

// A tensor of computational atoms: '00'
// CHECK: qwerty.t = #qwerty.vectree<VectorTensor [#qwerty.vectree<ZeroVector []>, #qwerty.vectree<ZeroVector []>]>
module attributes {qwerty.t = #qwerty.vectree<VectorTensor [#qwerty.vectree<ZeroVector []>, #qwerty.vectree<ZeroVector []>]>} {}

// -----

// The non-Pauli vector '0'@45 + '1'.
// CHECK: qwerty.t = #qwerty.vectree<UniformVectorSuperpos [#qwerty.vectree<VectorTilt tilt {{[0-9.eE+-]+}} : f64 [#qwerty.vectree<ZeroVector []>]>, #qwerty.vectree<OneVector []>]>
module attributes {qwerty.t = #qwerty.vectree<UniformVectorSuperpos [#qwerty.vectree<VectorTilt tilt 45.0 : f64 [#qwerty.vectree<ZeroVector []>]>, #qwerty.vectree<OneVector []>]>} {}
