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

    // rotl swaps
    for i in 0 .. (n - 2) {
	SWAP(qubits[i], qubits[i + 1]);
    }

    // Final single-qubit phase of 25 degrees on the target (qubit 0)
    let finalAngle = PI() * 25.0 / 180.0;
    R1(finalAngle, qubits[n - 1]);


    // Measure all qubits in Z and return the results as an array
    return MeasureEachZ(qubits);
}
