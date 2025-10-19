open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Convert;
open Microsoft.Quantum.Math;
open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Unstable.StatePreparation;

operation Fourier(n : Int) : Result[] {
    use qubits = Qubit[n];

    mutable statevector = [ComplexPolar(0.0, 0.0), size=1 <<< n];

    // Use all ones (the last fourier[N] basis vector)
    let j = ~~~(-1 <<< n);

    // Equation (5.2) of Nielsen and Chuang
    for k in 0..Length(statevector)-1 {
        let phase = 2.0*PI()*IntAsDouble(j)*IntAsDouble(k)/IntAsDouble(1 <<< n);
        set statevector w/= k <- ComplexPolar(Sqrt(1.0/IntAsDouble(1 <<< n)), phase);
    }

    ApproximatelyPreparePureStateCP(0.0, statevector, qubits);

    return MeasureEachZ(qubits);
}
