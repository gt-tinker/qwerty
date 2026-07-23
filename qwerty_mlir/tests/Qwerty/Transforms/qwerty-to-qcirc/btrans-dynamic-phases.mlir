// RUN: qwerty-opt -split-input-file -convert-qwerty-to-qcirc -verify-diagnostics %s

// Dynamic basis phases -- an angle supplied at runtime through a phases()
// operand, i.e. a dynamic (angle-less) VectorTilt in the vector tree -- are
// deferred. The conversion pass must refuse them with a clean legalization
// error rather than dropping or misrouting the operand. Reactivation recipe:
// docs/recursive-basis-vector-plan.md (v3 appendix).

qwerty.func @dynPhaseRefused[](%arg0: !qwerty<qbundle[1]>) irrev-> !qwerty<qbundle[1]> {
  %cst = arith.constant 3.1415926535897931 : f64
  // expected-error @below {{failed to legalize operation 'qwerty.qbtrans'}}
  %0 = qwerty.qbtrans %arg0 by {std:Z[1]} >> {list:{<VectorTilt tilt theta [#qwerty.vectree<OneVector []>]>, <ZeroVector []>}} phases (%cst) : (f64, !qwerty<qbundle[1]>) -> !qwerty<qbundle[1]>
  qwerty.return %0 : !qwerty<qbundle[1]>
}

