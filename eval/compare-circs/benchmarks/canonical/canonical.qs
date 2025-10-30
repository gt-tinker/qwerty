open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation Canonical(n : Int) : Result[] {
    use qubits = Qubit[n];

    // First, apply the Hadamard to the first qubit
    H(qubits[0]);

    // Then for each controlling qubit i = 1 .. n-1
    // apply a controlled phase rotation with angle pi / 2^i
    for i in 1 .. n - 1 {
	let control = qubits[i];
	let target = qubits[0];
	let angle = PI() / (2.0 ^ IntAsDouble(i));
	Controlled R1([control], (angle, target));
    }

    return MeasureEachZ(qubits);
}
