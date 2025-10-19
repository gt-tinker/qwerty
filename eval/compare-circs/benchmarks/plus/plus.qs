open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation Plus(n : Int) : Result[] {
    use qubits = Qubit[n];

    let statevector = [ComplexPolar(Sqrt(1.0/IntAsDouble(1 <<< n)), 0.0), size=1 <<< n];
    ApproximatelyPreparePureStateCP(0.0, statevector, qubits);

    return MeasureEachZ(qubits);
}
