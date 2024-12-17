open Microsoft.Quantum.Arrays;
open Microsoft.Quantum.Canon;
open Microsoft.Quantum.Measurement;
open Microsoft.Quantum.Intrinsic;

operation SimonAlgorithm(n : Int, oracle : ((Qubit[], Qubit[]) => Unit)) : Result[] {
    use (x, y) = (Qubit[n], Qubit[n]);
    ApplyToEach(H, x);
    oracle(x, y);
    ApplyToEach(H, x);
    ResetAll(y);
    return MResetEachZ(x);
}

// Hand-implemented version of the Simon's oracle in the Qwerty code
operation SecretSauceOracle(x : Qubit[], y : Qubit[]) : Unit {
    let n = Length(x);
    let k = n / 2;

    // Step 1: Copy the first n/2 input bits to the first n/2 output bits
    for i in 0..k-1 {
        CNOT(x[i], y[i]);
    }

    for i in k+1..n-1 {
        // Step 2: Copy the last n/2-1 input bits to the respective output bits
        CNOT(x[i], y[i]);
        // Step 3: XOR the kth input bit in-place with each of the last n/2-1
        //         output bits
        CNOT(x[k], y[i]);
    }
}
