open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation RevolvePhases(n : Int) : Result[] {
    use qubits = Qubit[n];

    // Prepare each qubit: apply S† (Adjoint S) then H
    for i in 0 .. n - 1 {
        Adjoint S(qubits[i]);
        H(qubits[i]);
    }

    H(qubits[0]);

    // For i = 1 .. n-1 apply a controlled R1(angle) with control qubits[i] -> target qubits[0]
    for i in 1 .. n - 1 {
        let control = qubits[i];
        let target = qubits[0];
	let angle = PI() / (2.0 ^ IntAsDouble(i));
        Controlled R1([control], (angle, target));
    }

    // TODO: Add swaps here

    // Final single-qubit phase of 25 degrees on the target (qubit 0)
    let finalAngle = PI() * 25.0 / 180.0;
    // R1(finalAngle, qubits[0]);
    Rz(finalAngle, qubits[0]);


    // Add swaps after RZ
    for i in 0 .. (n / 2 - 1) {
        SWAP(qubits[i], qubits[n - 1 - i]);
    }

    // Measure all qubits in Z and return the results as an array
    return MeasureEachZ(qubits);
}
