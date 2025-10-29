open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation QFT(n : Int) : Result[] {
    use qubits = Qubit[n];

    ApplyQFT(qubits);

    return MeasureEachZ(qubits);
}
