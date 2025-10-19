open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation RandPrep(n : Int, vec: ComplexPolar[]) : Result[] {
    use qubits = Qubit[n];
    ApproximatelyPreparePureStateCP(0.0, vec, qubits);
    return MeasureEachZ(qubits);
}
