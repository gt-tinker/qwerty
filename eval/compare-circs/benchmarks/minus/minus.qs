open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation Minus(n : Int) : Result[] {
    use qubits = Qubit[n];

    mutable statevector = [ComplexPolar(0.0, 0.0), size=1 <<< n];

    for i in 0..Length(statevector)-1 {
        let bits = IntAsBoolArray(i, n);
        let parity = Fold((x, y) -> (x and not y) or (not x and y), false, bits);
        let phase = parity ? PI() | 0.0;
        set statevector w/= i <- ComplexPolar(Sqrt(1.0/IntAsDouble(1 <<< n)), phase);
    }

    ApproximatelyPreparePureStateCP(0.0, statevector, qubits);

    return MeasureEachZ(qubits);
}
