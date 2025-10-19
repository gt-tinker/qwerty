open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation GHZ(n : Int) : Result[] {
    use qubits = Qubit[n];

    mutable statevector = [ComplexPolar(0.0, 0.0), size=1 <<< n];
    set statevector w/= 0 <- ComplexPolar(Sqrt(0.5), 0.0);
    set statevector w/= 1 <- ComplexPolar(Sqrt(0.5), 0.0);

    ApproximatelyPreparePureStateCP(0.0, statevector, qubits);

    return MeasureEachZ(qubits);
}
